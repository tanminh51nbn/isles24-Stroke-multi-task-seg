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

def train_worker(rank: int, world_size: int, args):
    """
    Hàm worker chạy trên mỗi GPU.

    Args:
        rank:       GPU index (0 hoặc 1)
        world_size: Tổng số GPU (2)
        args:       Parsed CLI arguments
    """
    # Khởi tạo DDP process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # ── Chỉ GPU 0 in log — tắt stdout của các GPU còn lại ────────────────────
    import sys as _sys
    import io as _io
    if rank != 0:
        _sys.stdout = _io.StringIO()  # Redirect sang buffer rỗng, không in gì cả

    print("=" * 60)
    print(" ISLES'24 Dual-Encoder Multi-Task UNet")
    print(f" World size: {world_size} GPU(s)")
    print("=" * 60)

    # Load config
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    config = load_configs(config_dir)

    # Ghi đè đường dẫn weights nếu có truyền từ CLI
    if args.cta_weights:
        config["cta_encoder"]["weights"] = args.cta_weights
    if args.perf_weights:
        config["perfusion_encoder"]["weights"] = args.perf_weights
    if args.metadata_path:
        config["sampling"]["metadata_csv"] = args.metadata_path

    # ── DataLoader ────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        config=config,
        dataset_dir=args.dataset_dir,
        rank=rank,
        world_size=world_size,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = build_model(config)
    model = model.to(device)
    # find_unused_parameters=True: Bắt buộc khi encoder bị freeze
    # DDP cần biết một số parameter không nhận gradient (frozen) là intentional
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # ── Loss / Optimizer / Scheduler ──────────────────────────────
    loss_fn   = MultiTaskLoss(config).to(device)
    optimizer = build_optimizer(model.module, config)
    scheduler = build_scheduler(optimizer, config)

    # ── Callbacks ─────────────────────────────────────────────────
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping"]["patience"],
        min_delta=config["training"]["early_stopping"]["min_delta"],
    ) if config["training"]["early_stopping"]["enabled"] else None

    checkpoint = ModelCheckpoint(
        save_dir=os.path.join(output_dir, config["training"]["checkpoint"]["dir"]),
        config=config,
    ) if rank == 0 else None

    # ── Training ──────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        rank=rank,
    )

    history = trainer.fit(
        early_stopping=early_stopping,
        checkpoint=checkpoint,
    )

    # ── Post-training (chỉ rank 0) ────────────────────────────────
    if rank == 0:
        # Vẽ training curves
        curve_path = os.path.join(output_dir, "training_curves.png")
        plot_training_curves(history, save_path=curve_path)

        # In báo cáo kết quả cuối
        best = max(history, key=lambda h: h["composite"])
        print("\n" + "=" * 60)
        print(" TRAINING COMPLETE — BEST RESULTS")
        print(f"  Epoch:          {best['epoch']}")
        print(f"  Dice Lesion:    {best['dice_lesion']:.4f}")
        print(f"  Recall LVO:     {best['recall_lvo']:.4f}")
        print(f"  Dice CoW:       {best['dice_cow']:.4f}")
        print(f"  Composite:      {best['composite']:.4f}")
        print("=" * 60)

    dist.destroy_process_group()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ISLES'24 Training Pipeline")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/kaggle/input/isles24-stroke-dataset/ISLES24_NPY_Dataset",
        help="Đường dẫn đến thư mục chứa file .npy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/kaggle/working/outputs",
        help="Thư mục lưu checkpoint và visualization",
    )
    parser.add_argument(
        "--cta_weights",
        type=str,
        default=None,
        help="Đường dẫn tùy chỉnh tới trọng số ResNet50",
    )
    parser.add_argument(
        "--perf_weights",
        type=str,
        default=None,
        help="Đường dẫn tùy chỉnh tới trọng số DenseNet121",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="/kaggle/working/dataset_metadata.csv",
        help="Đường dẫn tới file metadata CSV (phục vụ sampling)",
    )
    args = parser.parse_args()


    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("Không tìm thấy GPU!")

    print(f"[PINELINE] Phát hiện {world_size} GPU")

    if world_size > 1:
        # Multi-GPU: spawn DDP workers
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        mp.spawn(
            train_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single GPU (fallback)
        train_worker(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
