"""
src.losses — ISLES'24 Loss Functions

Public API:
    MultiTaskLoss  → Per-task DiceFocal + weighted aggregation
    build_loss     → Factory: config dict → initialized loss module
"""
from .multi_task_loss import MultiTaskLoss, build_loss

__all__ = [
    "MultiTaskLoss",
    "build_loss",
]
