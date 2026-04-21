"""
src.engine — ISLES'24 Training Engine

Public API:
    Trainer          → Full training pipeline with Accelerate DDP
    build_optimizer  → AdamW with differential LR
    build_scheduler  → CosineAnnealing with linear warmup
"""
from .optim import build_optimizer, build_scheduler
from .trainer import Trainer

__all__ = [
    "Trainer",
    "build_optimizer",
    "build_scheduler",
]
