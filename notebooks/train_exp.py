# %% [markdown]
# # Experiment Runner: ISLES'24 2.5D Multi-Task Segmentation
# 
# Notebook này được thiết kế để chạy trực tiếp trên Kaggle (hoặc local). 
# Bao gồm **Exp 0 (Sanity check)** và **Exp 1 (Baseline)**.
# 
# Các cài đặt môi trường cần thiết (chạy lệnh này trong terminal nếu chưa cài):
# `pip install -q segmentation-models-pytorch accelerate wandb monai`

# %% [code]
import os
import sys
from pathlib import Path
import yaml
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
    # Clone vào /kaggle/working/
    os.chdir("/kaggle/working")
    if not Path(REPO_NAME).exists():
        os.system(f"git clone {REPO_URL}")
    os.chdir(REPO_NAME)
    os.system(f"git checkout {BRANCH}")
    os.system("git pull")
    workspace_dir = Path(os.getcwd())
else:
    # Local Setup
    workspace_dir = Path("d:/Document/isles24_seg")
    if not workspace_dir.exists():
        workspace_dir = Path("/content/drive/MyDrive/Dataset/Stroke/isles24_seg") # Google Colab fallback
    
if str(workspace_dir) not in sys.path:
    sys.path.append(str(workspace_dir))

print(f"Workspace set to: {workspace_dir}")

# %% [markdown]
# ## 1. Khởi tạo môi trường & Load Config

# %% [code]
import torch
from accelerate.utils import write_basic_config
from accelerate import notebook_launcher
import urllib.request

# Tự động tải RadImageNet ResNet50 Weights nếu chưa có
rad_weights_path = workspace_dir / "configs" / "RadImageNet-ResNet50.pt"
if not rad_weights_path.exists():
    print("⏳ Đang tải RadImageNet ResNet50 Weights (~90MB)...")
    url = "https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/ResNet50.pt"
    urllib.request.urlretrieve(url, str(rad_weights_path))
    print("✅ Tải hoàn tất RadImageNet Weights!")
else:
    print("✅ RadImageNet Weights đã có sẵn.")

# Load train config
config_path = workspace_dir / "configs/train.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Load model config
model_config_path = workspace_dir / "configs/model.yaml"
with open(model_config_path, "r", encoding="utf-8") as f:
    model_cfg = yaml.safe_load(f)

# Merge configs
cfg.update(model_cfg)
    
# Cập nhật đường dẫn tuyệt đối cho weights
if cfg["encoder"]["weights"] == "RadImageNet-ResNet50.pt":
    cfg["encoder"]["weights"] = str(rad_weights_path)

# Tắt WandB như yêu cầu
cfg["training"]["logging"]["wandb"]["enabled"] = False
print("📡 WandB has been DISABLED.")

# Override config cho Exp 0 (Sanity Check)
# Hãy comment đoạn này lại nếu muốn chạy Exp 1 (Baseline 50 Epochs)
print("--- RUNNING EXP 0 (SANITY CHECK) ---")
cfg["training"]["epochs"] = 2
cfg["training"]["batch_size"] = 2
# Nếu chạy Exp 1 thì epochs=50, batch_size=4

# %% [markdown]
# ## 2. Build Dataset & Visual Sanity Check
# Kiểm tra xem Dataloader có trả về đúng Shape và Brain Mask không.

# %% [code]
from src.data.dataset import ISLES24Dataset
import monai.transforms as mt

# Define Transforms
val_transform = mt.Compose([
    mt.EnsureChannelFirstd(keys=["image", "label", "brain_mask"], channel_dim="no_channel"),
    # Add any other validation transforms here
])

# ── Tìm kiếm dữ liệu trên Kaggle (Gộp Part 1 & Part 2) ──
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    part1 = Path("/kaggle/input/datasets/muynhmuynh/isles24-stroke-dataset-part-1")
    part2 = Path("/kaggle/input/datasets/muynhmuynh/isles24-stroke-dataset-part-2")
    
    patient_dirs = {}
    for d in [part1, part2]:
        if d.exists():
            # Tìm tất cả thư mục sub-* trong cả 2 part
            patient_dirs.update({p.name: p for p in d.glob("sub-*") if p.is_dir()})
    
    print(f"📦 Đã tìm thấy {len(patient_dirs)} bệnh nhân từ cả 2 Part.")
else:
    # Local path
    data_dir = workspace_dir / "data" / "processed" 
    patient_dirs = {p.name: p for p in data_dir.glob("sub-*") if p.is_dir()}

if not patient_dirs:
    print("⚠️ Không tìm thấy patient_dirs. Đảm bảo bạn đã map đúng đường dẫn Dataset trên Kaggle.")
else:
    patient_ids = list(patient_dirs.keys())
    train_ids = patient_ids[:int(len(patient_ids)*0.8)]
    
    # Init Dataset
    train_ds = ISLES24Dataset(
        patient_ids=train_ids,
        patient_dirs=patient_dirs,
        pos_weight=3.0 # LVO oversample theo design plan
    )
    
    # Lấy thử 1 batch
    x, y, mask = train_ds[0]
    print(f"Input X shape: {x.shape} (Dtype: {x.dtype})")
    print(f"Label Y shape: {y.shape} (Dtype: {y.dtype})")
    print(f"Brain Mask shape: {mask.shape} (Dtype: {mask.dtype})")
    print(f"Perfusion CBF range (Channel 9): [{x[9].min():.2f}, {x[9].max():.2f}] - Should be clipped to [-5, 5]")

# %% [markdown]
# ## 3. Hàm Training Chính (Dành cho Accelerate)
# Bọc toàn bộ logic khởi tạo Model, Loss, Trainer vào một hàm để `notebook_launcher` gọi trên các tiến trình GPU.

# %% [code]
def training_function(cfg, patient_dirs):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    from src.models.network import build_model
    from src.losses.multi_task_loss import MultiTaskLoss
    from src.engine.trainer import MultiTaskTrainer
    from src.data.dataset import ISLES24Dataset
    
    # 1. Dataset & Dataloader
    patient_ids = sorted(list(patient_dirs.keys()))
    split_idx = int(len(patient_ids) * 0.8)
    train_ids = patient_ids[:split_idx]
    val_ids = patient_ids[split_idx:]
    
    train_ds = ISLES24Dataset(patient_ids=train_ids, patient_dirs=patient_dirs)
    val_ds = ISLES24Dataset(patient_ids=val_ids, patient_dirs=patient_dirs)
    
    # Weighted Sampler
    weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg["training"]["batch_size"], 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg["training"]["batch_size"], 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 2. Model & Loss
    model = build_model(cfg)
    criterion = MultiTaskLoss(cfg)
    
    # 3. Optimizer & Scheduler
    # Warmup + Cosine Annealing (Design Plan Khuyến nghị)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=1e-6
    )
    
    # 4. Initialize Trainer & Run
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

# %% [markdown]
# ## 4. Kích hoạt Training qua Accelerate
# Lệnh dưới đây sẽ chạy DDP (Distributed Data Parallel) một cách tự động tùy thuộc vào số lượng GPU Kaggle cung cấp (T4 x2 chẳng hạn).

# %% [code]
if __name__ == "__main__":
    if patient_dirs:
        print("Bắt đầu tiến trình Training...")
        # num_processes = số lượng GPU. 
        # Nếu dùng 1 GPU (như P100/A100), để num_processes=1
        notebook_launcher(training_function, args=(cfg, patient_dirs), num_processes=1)
    else:
        print("Dừng: Không tìm thấy dữ liệu.")
