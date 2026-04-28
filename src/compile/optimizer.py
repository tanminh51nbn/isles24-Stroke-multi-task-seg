"""
optimizer.py — AdamW với Differential Learning Rate

Chiến lược Differential LR:
    - Encoder LR = base_lr × encoder_lr_scale (mặc định 0.1)
      → Encoder đã có RadImageNet weights tốt, chỉ fine-tune nhẹ nhàng
      → Tránh "quên" kiến thức y tế đã học (Catastrophic Forgetting)
    - Decoder + Heads LR = base_lr
      → Đây là phần train từ đầu, cần học nhanh
"""

import torch
from torch.optim import AdamW


def build_optimizer(model, config: dict) -> AdamW:
    """
    Xây dựng AdamW với 2 param groups (Differential LR).

    Args:
        model:  DualEncoderUNet instance
        config: Dict đọc từ train.yaml

    Returns:
        AdamW optimizer với 2 lr groups
    """
    opt_cfg = config["optimizer"]
    base_lr  = float(opt_cfg["lr"])
    enc_lr   = base_lr * float(opt_cfg["encoder_lr_scale"])
    wd       = float(opt_cfg["weight_decay"])

    param_groups = model.get_param_groups(
        encoder_lr=enc_lr,
        decoder_lr=base_lr,
    )

    optimizer = AdamW(param_groups, weight_decay=wd)

    print(f"[Optimizer] AdamW | Encoder LR: {enc_lr:.2e} | Decoder LR: {base_lr:.2e} | WD: {wd:.2e}")
    return optimizer
