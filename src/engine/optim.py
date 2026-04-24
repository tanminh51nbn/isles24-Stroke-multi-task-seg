import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """
    Builds the AdamW optimizer with differential learning rates for Encoder vs Heads.
    """
    opt_cfg = cfg.get("optimizer", {})
    lr = opt_cfg.get("lr", 3e-4)
    encoder_lr_scale = opt_cfg.get("encoder_lr_scale", 0.1)
    weight_decay = opt_cfg.get("weight_decay", 1e-4)
    
    # Separate parameters
    encoder_params = []
    other_params = []
    
    # Lấy module thực tế nếu đang dùng DDP
    actual_model = model.module if hasattr(model, "module") else model

    for name, param in actual_model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            encoder_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": lr * encoder_lr_scale},
            {"params": other_params, "lr": lr}
        ],
        weight_decay=weight_decay
    )
    return optimizer

def build_scheduler(optimizer, cfg):
    train_cfg = cfg.get("training", {})
    sched_cfg = cfg.get("scheduler", {})
    
    epochs = train_cfg.get("epochs", 50)
    warmup_epochs = sched_cfg.get("warmup_epochs", 5)
    min_lr = sched_cfg.get("min_lr", 1e-6)

    # 1. Linear Warmup
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.001, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    # 2. Cosine Decay
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs, 
        eta_min=min_lr
    )
    
    # 3. Combine using SequentialLR
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    return scheduler
