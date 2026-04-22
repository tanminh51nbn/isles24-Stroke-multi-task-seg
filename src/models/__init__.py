"""
src.models — ISLES'24 Multi-Task Segmentation Models

Public API:
    MultiTaskSharedUNet → Multi-task 2.5D Shared U-Net (shared encoder, shared decoder)
    build_model         → Factory: config dict → initialized model
"""
from .network import MultiTaskSharedUNet, build_model

__all__ = [
    "MultiTaskSharedUNet",
    "build_model",
]
