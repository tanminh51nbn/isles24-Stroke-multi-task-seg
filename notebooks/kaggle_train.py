# %% [markdown]
# # ISLES'24 Multi-Task Segmentation — Kaggle Training Script
# Hardware: GPU T4 x2 | Framework: PyTorch + Accelerate DDP

# %% [markdown]
# ## Cell 1: Environment Setup
# Run this cell ONCE to install libraries and download code.
# ⚠️ If you change code on GitHub, Restart Kernel and run this cell again.

# %% [code]
!pip install -q -U monai segmentation-models-pytorch accelerate wandb
!rm -rf /kaggle/working/isles24_seg
!git clone https://github.com/tanminh51nbn/isles24-Stroke-multi-task-seg.git /kaggle/working/isles24_seg

# %% [markdown]
# ## Cell 2: Launch Multi-GPU Training
# ⚠️ CRITICAL RULE: In Kaggle, you CANNOT import `torch` or any of your `src` modules 
# in the main notebook cells before calling `notebook_launcher`. Doing so will initialize 
# CUDA in the main process, causing `notebook_launcher` (which forks the process) to crash 
# with the error: "Cannot re-initialize CUDA in forked subprocess".
# 
# Therefore, EVERYTHING (imports, data, model) must be INSIDE `training_function`.

# %% [code]
def training_function():
    """
    Self-contained training function for Accelerate DDP.
    Everything must be imported inside here!
    """
    # ── 0. PATH SETUP & DDP FIX (Critical for spawned processes) ──
    import sys, os
    
    # ❌ Fix Kaggle NCCL Hang: Disable P2P and InfiniBand
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    project_path = "/kaggle/working/isles24_seg"
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
    os.chdir(project_path)

    # ── 1. DEPENDENCY IMPORTS ──
    import yaml
    import torch
    import gc
    from pathlib import Path

    from src.data import (build_kfold_splits, ISLES24Dataset,
                          build_train_transforms, build_val_transforms,
                          build_dataloaders)
    from src.models import build_model
    from src.losses import build_loss
    from src.engine import Trainer, build_optimizer, build_scheduler

    # ── 2. CLEAR VRAM ──
    torch.cuda.empty_cache()
    gc.collect()

    # ── 3. LOAD CONFIGS ──
    project_dir = Path(project_path)
    data_cfg  = yaml.safe_load(open(project_dir / "configs/data.yaml"))
    model_cfg = yaml.safe_load(open(project_dir / "configs/model.yaml"))
    train_cfg = yaml.safe_load(open(project_dir / "configs/train.yaml"))

    # Force checkpoint path to /kaggle/working for easy download
    train_cfg["training"]["checkpoint"]["dir"] = "/kaggle/working/checkpoints"

    # ── SECURE W&B LOGIN ──
    try:
        from kaggle_secrets import UserSecretsClient
        import wandb
        user_secrets = UserSecretsClient()
        wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
        wandb.login(key=wandb_api_key)
        
        # Turn it on in config dynamically
        if "logging" not in train_cfg["training"]:
            train_cfg["training"]["logging"] = {}
        if "wandb" not in train_cfg["training"]["logging"]:
            train_cfg["training"]["logging"]["wandb"] = {}
            
        train_cfg["training"]["logging"]["wandb"]["enabled"] = True
        print("✅ Thành công kết nối W&B!")
    except Exception as e:
        print("⚠️ W&B tắt do không cấu hình Kaggle Secrets.")
        try:
            train_cfg["training"]["logging"]["wandb"]["enabled"] = False
        except KeyError:
            pass

    # ── 4. DATA PIPELINE ──
    data_dirs = [
        "/kaggle/input/datasets/muynhmuynh/isles24-stroke-dataset-part-1",
        "/kaggle/input/datasets/muynhmuynh/isles24-stroke-dataset-part-2"
    ]

    splits, patient_dirs = build_kfold_splits(
        data_dirs, data_cfg["kfold"], cache_dir="/kaggle/working"
    )

    train_ids, val_ids = splits[0] # Training Fold 0

    train_ds = ISLES24Dataset(
        train_ids, patient_dirs,
        transform=build_train_transforms(data_cfg),
        cache_path="/kaggle/working/cache_train_f0.json",
        pos_weight=data_cfg["sampling"]["pos_weight"],
    )

    val_ds = ISLES24Dataset(
        val_ids, patient_dirs,
        transform=build_val_transforms(),
        cache_path="/kaggle/working/cache_val_f0.json",
    )

    train_loader, val_loader = build_dataloaders(train_ds, val_ds, data_cfg)

    # ── 5. MODEL, LOSS, OPTIMIZER ──
    model     = build_model(model_cfg)
    criterion = build_loss(train_cfg)
    optimizer = build_optimizer(model, train_cfg["optimizer"])
    scheduler = build_scheduler(
        optimizer, train_cfg["scheduler"], train_cfg["training"]["epochs"]
    )

    # ── 6. TRAIN ──
    trainer = Trainer(
        model, criterion, optimizer, scheduler,
        train_loader, val_loader, train_cfg,
    )
    trainer.fit()


# ── THE ONLY THING ALLOWED IN THE GLOBAL SCOPE ──
from accelerate import notebook_launcher

if __name__ == "__main__":
    # Launch DDP with 2 GPUs
    notebook_launcher(training_function, num_processes=2)
