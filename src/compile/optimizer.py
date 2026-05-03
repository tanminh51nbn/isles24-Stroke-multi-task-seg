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


def build_optimizer(model, loss_fn, config: dict) -> AdamW:
    """
    Xây dựng AdamW với 2 param groups (Differential LR) và Uncertainty Weights.

    Args:
        model:   DualEncoderUNet instance
        loss_fn: MultiTaskLoss instance chứa Uncertainty weights
        config:  Dict đọc từ train.yaml

    Returns:
        AdamW optimizer với 3 lr groups
    """
    opt_cfg = config["optimizer"]
    base_lr  = float(opt_cfg["lr"])
    enc_lr   = base_lr * float(opt_cfg["encoder_lr_scale"])
    wd       = float(opt_cfg["weight_decay"])
    eps      = float(opt_cfg.get("eps", 1e-8))

    param_groups = model.get_param_groups(
        encoder_lr=enc_lr,
        decoder_lr=base_lr,
    )

    # Thêm tham số Uncertainty Weighting (log_vars) vào optimizer
    if hasattr(loss_fn, "parameters") and len(list(loss_fn.parameters())) > 0:
        param_groups.append({
            "params": list(loss_fn.parameters()),
            "lr": base_lr * 5.0, # Uncertainty cần hội tụ nhanh hơn một chút
            "name": "uncertainty_weights",
            "weight_decay": 0.0 # Không áp dụng weight decay cho tham số này
        })

    optimizer = AdamW(param_groups, weight_decay=wd, eps=eps)

    print(f"[Optimizer] AdamW | Encoder LR: {enc_lr:.2e} | Decoder LR: {base_lr:.2e} | WD: {wd:.2e} | Eps: {eps:.1e}")
    return optimizer
