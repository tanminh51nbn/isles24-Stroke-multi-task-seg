"""
optim.py — Optimizer & LR Scheduler factories for ISLES'24.

Optimizer: AdamW with differential learning rates
  - Encoder:        lr × encoder_lr_scale (protect pretrained features)
  - Decoders+Heads: lr (full learning rate, random init)

Scheduler: Linear Warmup → Cosine Annealing (pure PyTorch)
  - Warmup:  0 → lr over warmup_epochs (safe cold start)
  - Cosine:  lr → min_lr over remaining epochs (fine convergence)
"""
import logging

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

logger = logging.getLogger(__name__)


def build_optimizer(model, cfg: dict) -> optim.Optimizer:
    """
    Build AdamW optimizer with differential learning rates.

    Encoder params use a scaled-down LR to protect pretrained ImageNet
    features from catastrophic forgetting. Decoder/head params use full LR
    since they are randomly initialized.

    Args:
        model: MultiTaskUNet instance
        cfg:   optimizer section from configs/train.yaml

    Returns:
        AdamW optimizer with 2 param groups
    """
    lr = cfg["lr"]                                      # 3e-4
    weight_decay = cfg.get("weight_decay", 1e-4)
    encoder_lr_scale = cfg.get("encoder_lr_scale", 0.1)

    encoder_lr = lr * encoder_lr_scale                  # 3e-5

    # ── Separate encoder vs decoder/head parameters ──
    encoder_params = []
    decoder_head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            decoder_head_params.append(param)

    param_groups = [
        {
            "params": encoder_params,
            "lr": encoder_lr,
            "name": "encoder",
        },
        {
            "params": decoder_head_params,
            "lr": lr,
            "name": "decoders_heads",
        },
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

    logger.info(
        f"AdamW optimizer: "
        f"encoder LR={encoder_lr:.1e} ({len(encoder_params)} tensors), "
        f"decoders+heads LR={lr:.1e} ({len(decoder_head_params)} tensors), "
        f"weight_decay={weight_decay:.1e}"
    )

    return optimizer


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: dict,
    total_epochs: int,
) -> SequentialLR:
    """
    Build Cosine Annealing with Linear Warmup scheduler.

    Schedule (per epoch):
        Epoch 0→5:   Linear warmup    (lr×0.001 → lr)
        Epoch 5→50:  Cosine annealing  (lr → min_lr)

    Uses PyTorch built-in SequentialLR — no extra dependencies.

    Args:
        optimizer:    Configured optimizer
        cfg:          scheduler section from configs/train.yaml
        total_epochs: Total training epochs

    Returns:
        SequentialLR scheduler (call .step() once per epoch)
    """
    warmup_epochs = cfg.get("warmup_epochs", 5)
    min_lr = cfg.get("min_lr", 1e-6)

    # ── Phase 1: Linear Warmup ──
    # LR ramps from (start_factor × base_lr) to base_lr
    warmup = LinearLR(
        optimizer,
        start_factor=0.001,          # Start at 0.1% of base LR
        end_factor=1.0,              # End at 100% of base LR
        total_iters=warmup_epochs,
    )

    # ── Phase 2: Cosine Annealing ──
    # LR decays from base_lr to min_lr following cosine curve
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr,
    )

    # ── Chain: Warmup → Cosine ──
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )

    logger.info(
        f"LR Scheduler: LinearWarmup({warmup_epochs} epochs) → "
        f"CosineAnnealing({total_epochs - warmup_epochs} epochs, "
        f"min_lr={min_lr:.1e})"
    )

    return scheduler
