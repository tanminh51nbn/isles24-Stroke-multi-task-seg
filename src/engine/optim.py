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

def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    """
    Builds the Cosine Annealing scheduler.
    """
    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 50)
    sched_cfg = cfg.get("scheduler", {})
    min_lr = sched_cfg.get("min_lr", 1e-6)
    
    # Single cosine cycle over the full training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=min_lr
    )
    return scheduler
