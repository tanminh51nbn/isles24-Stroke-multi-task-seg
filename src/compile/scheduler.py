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
    Xây dựng scheduler: Linear Warmup → CosineAnnealing.

    Args:
        optimizer:       AdamW optimizer
        config:          Dict đọc từ train.yaml
        steps_per_epoch: Số batch mỗi epoch (dùng nếu muốn step-level scheduling)

    Returns:
        SequentialLR (warmup → cosine)
    """
    sched_cfg   = config["scheduler"]
    train_cfg   = config["training"]
    warmup_ep   = int(sched_cfg["warmup_epochs"])
    total_ep    = int(train_cfg["epochs"])
    min_lr      = float(sched_cfg["min_lr"])

    # --- Warmup: Linear tăng từ 0 → 1 (× base_lr) ---
    def warmup_lambda(epoch):
        if epoch < warmup_ep:
            return float(epoch + 1) / float(warmup_ep)
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # --- Cosine Annealing sau warmup ---
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_ep - warmup_ep,
        eta_min=min_lr,
    )

    # Kết hợp: warmup_ep epoch đầu dùng warmup, còn lại dùng cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_ep],
    )

    print(f"[Scheduler] Warmup {warmup_ep} epochs → CosineAnnealing {total_ep - warmup_ep} epochs (min_lr={min_lr:.2e})")
    return scheduler
