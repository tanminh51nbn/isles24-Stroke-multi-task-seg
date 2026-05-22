"""
scheduler.py — CosineAnnealing với Linear Warmup

Lý do cần Warmup:
    - 5 epoch đầu encoder bị freeze → decoder LR cao nhưng encoder không update
    - Sau khi unfreeze encoder, gradient đột ngột xuất hiện từ encoder
    - Nếu không có warmup → gradient explosion ở epoch unfreeze

Chiến lược:
    Epoch 0   → LR: 0 (bắt đầu warmup)
    Epoch W   → LR: base_lr (kết thúc warmup, bắt đầu Cosine decay)
    Epoch End → LR: min_lr
"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR


def build_scheduler(optimizer, config: dict, steps_per_epoch: int = 1):
    """
    Xây dựng scheduler 3 giai đoạn: Linear Warmup → Constant Hold → CosineAnnealing.

    Args:
        optimizer:       AdamW optimizer
        config:          Dict đọc từ train.yaml
        steps_per_epoch: Số batch mỗi epoch

    Returns:
        SequentialLR (warmup → hold → cosine)
    """
    sched_cfg   = config["scheduler"]
    train_cfg   = config["training"]
    warmup_ep   = int(sched_cfg["warmup_epochs"])
    hold_ep     = int(sched_cfg.get("hold_epochs", 0))
    total_ep    = int(train_cfg["epochs"])
    min_lr      = float(sched_cfg["min_lr"])

    # 1. --- Warmup: Linear tăng từ 0 → 1 (× base_lr) ---
    def warmup_lambda(epoch):
        return float(epoch + 1) / float(warmup_ep)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # 2. --- Hold: Giữ nguyên base_lr (Constant LR) ---
    # LambdaLR trả về constant 1.0 (nhân với base_lr)
    hold_scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    # 3. --- Cosine Annealing sau giai đoạn Hold ---
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_ep - warmup_ep - hold_ep,
        eta_min=min_lr,
    )

    # Kết hợp 3 giai đoạn bằng SequentialLR
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, hold_scheduler, cosine_scheduler],
        milestones=[warmup_ep, warmup_ep + hold_ep],
    )

    print(
        f"[Scheduler] Phased Strategy: Warmup({warmup_ep}ep) "
        f"-> Hold({hold_ep}ep) -> Cosine({total_ep - warmup_ep - hold_ep}ep) | "
        f"min_lr={min_lr:.2e}"
    )
    return scheduler
