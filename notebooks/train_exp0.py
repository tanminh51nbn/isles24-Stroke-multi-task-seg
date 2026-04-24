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

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    print("🚀 Đang chạy trên Kaggle. Đang thiết lập mã nguồn...")
    os.chdir("/kaggle/working")
    if not Path(REPO_NAME).exists():
        os.system(f"git clone {REPO_URL}")
    os.chdir(REPO_NAME)
    os.system(f"git checkout {BRANCH}")
    os.system("git pull")
    workspace_dir = Path(os.getcwd())
else:
    # Local Setup
    workspace_dir = Path.cwd()

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
# 4. HÀM CHẠY THÍ NGHIỆM CHÍNH
def run_experiment(exp_name="Baseline_ResNet50"):
    print(f"🚀 Bắt đầu thí nghiệm: {exp_name}")
    
    # --- A. Load Configs ---
    workspace_dir = Path.cwd()
    if "notebooks" in str(workspace_dir): workspace_dir = workspace_dir.parent
    
    with open(workspace_dir / "configs/train.yaml", "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    with open(workspace_dir / "configs/model.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(workspace_dir / "configs/data.yaml", "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
        
    cfg = {**train_cfg, **model_cfg, **data_cfg}
    
    # Download weights
    rad_path = workspace_dir / "configs" / "RadImageNet-ResNet50.pt"
    download_radimagenet(rad_path)
    cfg["encoder"]["weights"] = str(rad_path)

    # --- B. Tìm dữ liệu (Kaggle Support) ---
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        # Đường dẫn mặc định khi add dataset trên Kaggle
        data_roots = [
            Path("/kaggle/input/isles24-stroke-dataset-part-1"),
            Path("/kaggle/input/isles24-stroke-dataset-part-2")
        ]
    else:
        data_roots = [workspace_dir / "data/processed"]

    patient_dirs = {}
    for root in data_roots:
        if root.exists():
            patient_dirs.update({p.name: p for p in root.glob("sub-*") if p.is_dir()})
    
    patient_ids = sorted(list(patient_dirs.keys()))
    print(f"📦 Tổng số bệnh nhân tìm thấy: {len(patient_ids)}")
    
    # --- C. Data Split (Simple 80/20 for baseline) ---
    split_idx = int(len(patient_ids) * 0.8)
    train_ids = patient_ids[:split_idx]
    val_ids = patient_ids[split_idx:]
    
    train_ds = ISLES24Dataset(patient_ids=train_ids, patient_dirs=patient_dirs)
    val_ds = ISLES24Dataset(patient_ids=val_ids, patient_dirs=patient_dirs)
    
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
    eval_patient_dirs = [patient_dirs[pid] for pid in val_ids]
    evaluator = Evaluator(
        model=model,
        patient_dirs=eval_patient_dirs,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    final_metrics = evaluator.evaluate_all()
    print(f"✅ Kết quả cuối cùng: {final_metrics}")

# %% [code]
if __name__ == "__main__":
    # Bạn có thể ghi đè tham số tại đây nếu cần (ví dụ: epochs=1 để test)
    run_experiment()