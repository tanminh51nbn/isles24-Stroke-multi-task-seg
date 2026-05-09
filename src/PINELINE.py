"""
PINELINE.py — Entry point cho toàn bộ training pipeline

Thứ tự thực thi:
    1. Load config từ YAML
    2. Khởi tạo DDP (DistributedDataParallel) trên 2 GPU T4
    3. Build DataLoader (train/val, patient-level split)
    4. Build DualEncoderUNet + Load RadImageNet weights
    5. Build Loss + Optimizer + Scheduler
    6. Build Callbacks (EarlyStopping, ModelCheckpoint)
    7. Chạy Trainer.fit()
    8. Vẽ Training Curves
    9. In báo cáo kết quả cuối cùng

Cách dùng trên Kaggle:
    python PINELINE.py \
        --dataset_dir /kaggle/input/isles24-stroke-dataset/ISLES24_NPY_Dataset \
        --output_dir  /kaggle/working/outputs
"""

import os
# [OPTIMIZE] Tránh phân mảnh bộ nhớ trên card T4
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Thêm src vào sys.path để import các module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models  import build_model
from data    import build_dataloaders
from compile import MultiTaskLoss, build_optimizer, build_scheduler
from training import Trainer, EarlyStopping, ModelCheckpoint
from evaluation import plot_training_curves


# ─── Config Loader ────────────────────────────────────────────────────────────

def load_configs(config_dir: str) -> dict:
    """Nạp và gộp 3 file YAML thành 1 config dict."""
    def _load(name):
        path = os.path.join(config_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    cfg = {}
    cfg.update(_load("model.yaml"))
    cfg.update(_load("data.yaml"))
    cfg.update(_load("train.yaml"))
    return cfg


# ─── DDP Worker ───────────────────────────────────────────────────────────────

def train_worker(rank: int, world_size: int, args, fold_idx: int = 0):
    """
    Hàm worker chạy trên mỗi GPU — hỗ trợ K-Fold.

    Args:
        rank:       GPU index (0 hoặc 1)
        world_size: Tổng số GPU (2)
        args:       Parsed CLI arguments
        fold_idx:   Fold hiện tại (0-4)
    """
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    import sys as _sys
    import io as _io
    if rank != 0:
        _sys.stdout = _io.StringIO()

    # Load config
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    config = load_configs(config_dir)

    # Ghi đè đường dẫn từ CLI
    if args.cta_weights:
        config["cta_encoder"]["weights"] = args.cta_weights
    if args.perf_weights:
        config["perfusion_encoder"]["weights"] = args.perf_weights
    if args.metadata_path:
        config["sampling"]["metadata_csv"] = args.metadata_path

    # [FIX 3] Ghi đè fold hiện tại vào config
    config["split"]["current_fold"] = fold_idx

    # Output riêng cho từng fold
    fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_output_dir, exist_ok=True)

    if rank == 0:
        n_folds = config["split"].get("n_folds", 1)
        mode    = config["split"].get("mode", "single")
        print("=" * 60)
        print(f" ISLES'24 — Fold {fold_idx + 1}/{n_folds} (mode={mode})")
        print("=" * 60)

    # ── DataLoader ────────────────────────────────────────────────
    train_loader, val_loader, train_files_original = build_dataloaders(
        config=config,
        dataset_dir=args.dataset_dir,
        rank=rank,
        world_size=world_size,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = build_model(config)
    model = model.to(device)
    model = DDP(
        model, 
        device_ids=[rank], 
        output_device=rank,
        find_unused_parameters=True
    )

    # ── Loss / Optimizer / Scheduler ──────────────────────────────
    loss_fn   = MultiTaskLoss(config).to(device)
    optimizer = build_optimizer(model.module, loss_fn, config)
    scheduler = build_scheduler(optimizer, config)

    # ── Callbacks ─────────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping"]["patience"],
        min_delta=config["training"]["early_stopping"]["min_delta"],
        rank=rank
    ) if config["training"]["early_stopping"]["enabled"] else None

    checkpoint_callback = ModelCheckpoint(
        save_dir=os.path.join(fold_output_dir, config["training"]["checkpoint"]["dir"]),
        config=config,
        rank=rank
    ) if rank == 0 else None

    config["output_dir"] = fold_output_dir
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_files_original=train_files_original,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        rank=rank,
    )

    # ── [RESUME LOGIC] ───────────────────────────────────────────
    start_epoch = 0
    if args.resume_from:
        start_epoch = trainer.load_checkpoint(args.resume_from)

    # ── Fit ───────────────────────────────────────────────────────
    history = trainer.fit(
        early_stopping=early_stopping,
        checkpoint=checkpoint_callback,
        start_epoch=start_epoch
    )

    # ── Post-training (chỉ rank 0) ────────────────────────────────
    if rank == 0:
        curve_path = os.path.join(fold_output_dir, "training_curves.png")
        plot_training_curves(history, save_path=curve_path)

        import pandas as pd
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(fold_output_dir, "training_history.csv"), index=False)

        start_eval = config["training"]["checkpoint"].get("start_epoch", 1)
        relevant_history = [h for h in history if h["epoch"] >= start_eval] or history
        best = max(relevant_history, key=lambda h: h["composite"])

        print(f"\n{'='*60}")
        print(f" FOLD {fold_idx} COMPLETE — BEST (From Epoch {start_eval}+)")
        print(f"  Epoch:       {best['epoch']}")
        print(f"  Dice Lesion: {best['dice_lesion']:.4f}")
        print(f"  F1 LVO:      {best['f1_lvo']:.2f}%")
        print(f"  Dice CoW:    {best['dice_cow']:.4f}")
        print(f"  AAD (Area):  {best['aad_lesion']:.2f}%")
        print(f"  Composite:   {best['composite']:.4f}")
        print(f"{'='*60}")

    if rank == 0:
        # [FIX] Ép kiểu NumPy/Tensor về Python chuẩn để lưu JSON thành công
        best_serializable = {k: (v.item() if hasattr(v, 'item') else v) for k, v in best.items()}
        
        import json
        res_path = os.path.join(fold_output_dir, "best_result.json")
        with open(res_path, "w") as f:
            json.dump(best_serializable, f)
        best_result = best
    else:
        best_result = None

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    
    return best_result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ISLES'24 Training Pipeline")
    parser.add_argument("--dataset_dir", type=str,
        default="/kaggle/input/isles24-stroke-dataset/ISLES24_NPY_Dataset")
    parser.add_argument("--model_path", type=str, default="/kaggle/working/outputs/fold_0/checkpoints/best_overall.pt", help="Đường dẫn file .pt")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/outputs")
    parser.add_argument("--cta_weights",   type=str, default=None)
    parser.add_argument("--perf_weights",  type=str, default=None)
    parser.add_argument("--metadata_path", type=str,
        default="/kaggle/working/dataset_metadata.csv")
    parser.add_argument("--resume_from", type=str, default=None, 
        help="Đường dẫn file .pt để tiếp tục huấn luyện")
    parser.add_argument("--fold", type=int, default=None, 
        help="Chỉ chạy duy nhất 1 fold (0-4). Nếu để None sẽ chạy toàn bộ.")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("Không tìm thấy GPU!")
    print(f"[PINELINE] Phát hiện {world_size} GPU")

    # Đọc config để biết mode và n_folds
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    config = load_configs(config_dir)
    split_mode = config["split"].get("mode", "single")
    n_folds    = config["split"].get("n_folds", 1) if split_mode == "kfold" else 1

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    fold_results = []

    # Xác định danh sách fold cần chạy
    if args.fold is not None:
        folds_to_run = [args.fold]
    else:
        folds_to_run = list(range(n_folds))

    for fold_idx in folds_to_run:
        print(f"\n[PINELINE] ===== Bắt đầu Fold {fold_idx + 1} =====")

        # [FIX] Dùng port khác nhau cho mỗi fold để tránh "Address already in use"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(12355 + fold_idx)

        if world_size > 1:
            mp.spawn(
                train_worker,
                args=(world_size, args, fold_idx),
                nprocs=world_size,
                join=True,
            )
            # Sau khi spawn xong, đọc lại kết quả từ file (do rank 0 lưu)
            import json
            res_path = os.path.join(args.output_dir, f"fold_{fold_idx}", "best_result.json")
            if os.path.exists(res_path):
                with open(res_path, "r") as f:
                    result = json.load(f)
                fold_results.append({"fold": fold_idx, **result})
        else:
            result = train_worker(rank=0, world_size=1, args=args, fold_idx=fold_idx)
            if result:
                fold_results.append({"fold": fold_idx, **result})


    # In bảng tổng hợp kết quả tất cả fold
    if fold_results:
        import pandas as pd
        summary_df = pd.DataFrame(fold_results)
        summary_path = os.path.join(args.output_dir, "kfold_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*60}")
        print(f" K-FOLD SUMMARY ({n_folds} Folds)")
        print(f"{'='*60}")
        for r in fold_results:
            print(f"  Fold {r['fold']}: Dice_L={r['dice_lesion']:.4f}  "
                  f"F1_LVO={r['f1_lvo']:.2f}%  "
                  f"Dice_C={r['dice_cow']:.4f}  "
                  f"Composite={r['composite']:.4f}")
        print(f"{'─'*60}")
        print(f"  Mean:   Dice_L={summary_df['dice_lesion'].mean():.4f}  "
              f"F1_LVO={summary_df['f1_lvo'].mean():.2f}%  "
              f"Dice_C={summary_df['dice_cow'].mean():.4f}  "
              f"Composite={summary_df['composite'].mean():.4f}")
        print(f"  Std:    Dice_L={summary_df['dice_lesion'].std():.4f}  "
              f"F1_LVO={summary_df['f1_lvo'].std():.2f}%  "
              f"Dice_C={summary_df['dice_cow'].std():.4f}  "
              f"Composite={summary_df['composite'].std():.4f}")
        print(f"{'='*60}")
        print(f"[PINELINE] K-Fold summary saved to: {summary_path}")
        print(f"[PINELINE] Ensemble models nằm tại:")
        for i in range(n_folds):
            print(f"  fold_{i}/checkpoints/best_overall.pt")


if __name__ == "__main__":
    main()
