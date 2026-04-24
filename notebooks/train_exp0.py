# %% [markdown]
# # ISLES'24: Multi-Task 2.5D Stroke Segmentation
# **Backbone:** ResNet50 (RadImageNet) | **Tasks:** Lesion, LVO, CoW
# 
# This script is designed to run on Kaggle T4 GPUs. It uses `accelerate` for distributed training
# and implements the full clinical design plan (differential LR, shared decoder, brain masking).

# %% [code]
import os
import sys
from pathlib import Path

# 1. MÔI TRƯỜNG KAGGLE (Cài đặt dependencies)
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    print("🛠️ Đang cài đặt thư viện (SMP, Monai, Accelerate)...")
    os.system('pip install -q segmentation-models-pytorch monai accelerate pyyaml')

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════
#  Kaggle Setup: Clone & Checkout Branch
# ═══════════════════════════════════════════════════════════════
REPO_URL = "https://github.com/tanminh51nbn/isles24-Stroke-multi-task-seg.git"
BRANCH = "feat/resnet50-radimagenet"
REPO_NAME = "isles24-Stroke-multi-task-seg"

# --- A. Setup Environment ---
workspace_dir = Path.cwd()
# Nếu đang ở trong thư mục notebooks, lùi lại 1 cấp để ra root repo
if "notebooks" in str(workspace_dir):
    workspace_dir = workspace_dir.parent

if str(workspace_dir) not in sys.path:
    sys.path.append(str(workspace_dir))

# Import các module từ source
try:
    from src.data.dataset import ISLES24Dataset
    from src.models.model import build_model
    from src.losses.multitask import build_loss
    from src.engine.trainer import MultiTaskTrainer
    from src.engine.optim import build_optimizer, build_scheduler
    from src.metrics.evaluator import Evaluator
except ImportError as e:
    print(f"❌ Không thể import các module từ /src: {e}")

# %% [code]
# 3. TẢI PRETRAINED WEIGHTS (RADIMAGENET)
def download_radimagenet(target_path):
    import urllib.request
    if not target_path.exists():
        print("⏳ Đang tải RadImageNet ResNet50 Weights (~90MB)...")
        url = "https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/ResNet50.pt"
        urllib.request.urlretrieve(url, str(target_path))
        print("✅ Tải hoàn tất RadImageNet Weights!")
    else:
        print("✅ RadImageNet Weights đã có sẵn.")

# %% [code]
# 4. LOAD CONFIGS
workspace_dir = Path.cwd()
if "notebooks" in str(workspace_dir): workspace_dir = workspace_dir.parent

with open(workspace_dir / "configs/train.yaml", "r", encoding="utf-8") as f:
    train_cfg = yaml.safe_load(f)
with open(workspace_dir / "configs/model.yaml", "r", encoding="utf-8") as f:
    model_cfg = yaml.safe_load(f)
with open(workspace_dir / "configs/data.yaml", "r", encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)
    
cfg = {**train_cfg, **model_cfg, **data_cfg}

# %% [code]
# 5. HÀM CHẠY THÍ NGHIỆM CHÍNH
def run_experiment(cfg, exp_name="Baseline_ResNet50"):
    is_main = os.environ.get("RANK", "0") == "0"
    
    if is_main:
        print(f"🚀 Bắt đầu thí nghiệm: {exp_name}")
    
    workspace_dir = Path.cwd()
    if "notebooks" in str(workspace_dir): workspace_dir = workspace_dir.parent
    
    # Download weights
    rad_path = workspace_dir / "configs" / "RadImageNet-ResNet50.pt"
    if is_main:
        download_radimagenet(rad_path)
    cfg["encoder"]["weights"] = str(rad_path)

    # --- B. Tìm dữ liệu (Kaggle Support) ---
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        kaggle_input = Path("/kaggle/input")
        if is_main:
            print(f"🔍 Đang tự động quét dữ liệu trong {kaggle_input}...")
        # Tìm mọi thư mục bắt đầu bằng sub- trong toàn bộ thư mục input
        all_sub_dirs = list(kaggle_input.rglob("sub-*"))
        patient_dirs = {d.name: d for d in all_sub_dirs if d.is_dir()}
    else:
        data_root = workspace_dir / "data/processed"
        patient_dirs = {p.name: p for p in data_root.glob("sub-*") if p.is_dir()}

    patient_ids = sorted(list(patient_dirs.keys()))
    if is_main:
        print(f"📦 Tổng số bệnh nhân tìm thấy: {len(patient_ids)}")
    
    if len(patient_ids) == 0:
        print("❌ Dừng: Không tìm thấy dữ liệu bệnh nhân (sub-*). Hãy kiểm tra lại Dataset.")
        return

    # --- C. Data Split (Simple 80/20 for baseline) ---
    split_idx = int(len(patient_ids) * 0.8)
    train_ids = patient_ids[:split_idx]
    val_ids = patient_ids[split_idx:]
    
    # Chuyển đổi ID thành List[Path] đúng chuẩn cho ISLES24Dataset
    train_paths = [patient_dirs[pid] for pid in train_ids]
    val_paths = [patient_dirs[pid] for pid in val_ids]
    
    from src.data.dataset import build_dataset
    train_ds = build_dataset(train_paths, cfg, is_train=True)
    val_ds = build_dataset(val_paths, cfg, is_train=False)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg["training"].get("batch_size", 4),
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg["training"].get("batch_size", 4),
        shuffle=False, 
        num_workers=4
    )

    # --- D. Initialize Model, Loss, Opt ---
    model = build_model(cfg)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # --- E. Training ---
    trainer = MultiTaskTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg
    )
    
    trainer.train()

    # --- F. Final 3D Evaluation ---
    print("\n🏁 Đang tiến hành đánh giá 3D cuối cùng...")
    evaluator = Evaluator(
        model=model,
        patient_dirs=val_paths, # Truyền List[Path]
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    final_metrics = evaluator.evaluate_all()
    print(f"✅ Kết quả cuối cùng: {final_metrics}")

# %% [code]
from accelerate import notebook_launcher

if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════════
    #  EXP 0: SANITY CHECK CONFIG
    # ═══════════════════════════════════════════════════════════════
    cfg["training"]["epochs"] = 2
    cfg["training"]["batch_size"] = 8
    cfg["training"]["logging"]["visualize_every"] = 1
    
    # Kiểm tra môi trường chạy:
    # Nếu chạy qua lệnh `accelerate launch`, biến môi trường "RANK" sẽ tồn tại.
    if "RANK" in os.environ:
        run_experiment(cfg)
    else:
        # Nếu chạy trực tiếp trong Notebook cell
        try:
            from accelerate import notebook_launcher
            print("🚀 Đang khởi tạo qua notebook_launcher (2 GPU)...")
            notebook_launcher(run_experiment, args=(cfg,), num_processes=2)
        except Exception as e:
            print(f"⚠️ notebook_launcher không khả dụng hoặc lỗi, chạy 1 GPU mặc định.")
            run_experiment(cfg)