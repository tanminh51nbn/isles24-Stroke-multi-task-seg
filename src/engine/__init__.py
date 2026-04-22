"""
src.engine — ISLES'24 Training Engine

Public API:
    MultiTaskTrainer  → Full training pipeline with Accelerate DDP
    build_optimizer   → AdamW with differential LR
    build_scheduler   → CosineAnnealing with linear warmup
"""
from .optim import build_optimizer, build_scheduler
from .trainer import MultiTaskTrainer

__all__ = [
    "MultiTaskTrainer",
    "build_optimizer",
    "build_scheduler",
]
