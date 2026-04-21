"""
src.models — ISLES'24 Multi-Task Segmentation Models

Public API:
    MultiTaskUNet  → Multi-task 2.5D U-Net (shared encoder, 3 decoders)
    build_model    → Factory: config dict → initialized model
"""
from .network import MultiTaskUNet, build_model

__all__ = [
    "MultiTaskUNet",
    "build_model",
]
