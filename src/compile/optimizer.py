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
    Xây dựng AdamW với Differential Learning Rate cho Encoder và Decoder,
    Đồng thời loại bỏ weight decay cho bias và BatchNorm.

    Args:
        model:   DualEncoderUNet instance
        loss_fn: MultiTaskLoss instance
        config:  Dict đọc từ train.yaml

    Returns:
        AdamW optimizer với các param groups cấu hình weight decay có chọn lọc
    """
    opt_cfg = config["optimizer"]
    base_lr  = float(opt_cfg["lr"])
    enc_lr   = base_lr * float(opt_cfg["encoder_lr_scale"])
    wd       = float(opt_cfg["weight_decay"])
    eps      = float(opt_cfg.get("eps", 1e-8))

    raw_param_groups = model.get_param_groups(
        encoder_lr=enc_lr,
        decoder_lr=base_lr,
    )

    # Phân tách param groups để áp dụng weight decay chọn lọc
    param_groups = []
    for g in raw_param_groups:
        name = g.get("name", "params")
        lr = g["lr"]
        params = g["params"]

        decay_params = []
        no_decay_params = []
        for p in params:
            if not p.requires_grad:
                continue
            # Bias và BatchNorm/LayerNorm weights & biases thường có ndim < 2 (1D hoặc 0D)
            if p.ndim < 2:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        param_groups.append({
            "params": decay_params,
            "lr": lr,
            "weight_decay": wd,
            "name": f"{name}_decay"
        })
        param_groups.append({
            "params": no_decay_params,
            "lr": lr,
            "weight_decay": 0.0,
            "name": f"{name}_no_decay"
        })

    optimizer = AdamW(param_groups, eps=eps)

    print(f"[Optimizer] AdamW | Encoder LR: {enc_lr:.2e} | Decoder LR: {base_lr:.2e} | WD: {wd:.2e} (No-WD on bias/BN) | Eps: {eps:.1e}")
    return optimizer
