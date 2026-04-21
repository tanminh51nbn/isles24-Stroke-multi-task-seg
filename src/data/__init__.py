"""
src.data — ISLES'24 Data Pipeline

Public API:
    build_kfold_splits   → Stratified K-Fold at patient level
    ISLES24Dataset       → PyTorch Dataset with caching & sampling
    build_train_transforms / build_val_transforms → MONAI pipelines
    build_dataloaders    → DataLoader factory with WeightedRandomSampler
"""
from .dataset import ISLES24Dataset
from .fold_split import build_kfold_splits
from .transforms import build_train_transforms, build_val_transforms
from .dataloader import build_dataloaders

__all__ = [
    "ISLES24Dataset",
    "build_kfold_splits",
    "build_train_transforms",
    "build_val_transforms",
    "build_dataloaders",
]
