"""
dataloader.py — DataLoader factory with WeightedRandomSampler for
                balanced training on severely imbalanced medical data.

Train: WeightedRandomSampler → positive slices sampled proportionally more
Val:   Sequential → evaluate every sample exactly once, no shuffle
"""
import logging

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import ISLES24Dataset

logger = logging.getLogger(__name__)


def build_dataloaders(
    train_dataset: ISLES24Dataset,
    val_dataset: ISLES24Dataset,
    cfg: dict,
) -> tuple:
    """
    Build optimized Train & Val DataLoaders.

    Args:
        train_dataset: ISLES24Dataset for training (with augmentation)
        val_dataset:   ISLES24Dataset for validation (no augmentation)
        cfg:           Full data config dict (dataloader + sampling sections)

    Returns:
        (train_loader, val_loader)
    """
    dl_cfg = cfg.get("dataloader", {})
    samp_cfg = cfg.get("sampling", {})

    # ── Common DataLoader kwargs ──
    batch_size = dl_cfg.get("batch_size", 8)
    num_workers = dl_cfg.get("num_workers", 4)
    pin_memory = dl_cfg.get("pin_memory", True)
    persistent_workers = dl_cfg.get("persistent_workers", True) and num_workers > 0
    prefetch_factor = dl_cfg.get("prefetch_factor", 2) if num_workers > 0 else None

    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # ════════════════════════════════════════════════════════════
    #  Train DataLoader — WeightedRandomSampler
    # ════════════════════════════════════════════════════════════
    strategy = samp_cfg.get("strategy", "weighted")

    if strategy == "weighted":
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(train_dataset),   # 1 epoch = same # of iterations
            replacement=True,                  # Required for weighted sampling
        )
        n_pos = (weights > 1.0).sum()
        pos_w = samp_cfg.get("pos_weight", 3.0)
        logger.info(
            f"WeightedRandomSampler: {n_pos} positive slices (weight={pos_w}), "
            f"{len(weights) - n_pos} negative slices (weight=1.0)"
        )
    else:
        sampler = None
        logger.info("Using uniform random sampling (no weighting)")

    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=(sampler is None),       # Only shuffle if no custom sampler
        drop_last=dl_cfg.get("drop_last", True),
        **common_kwargs,
    )

    # ════════════════════════════════════════════════════════════
    #  Val DataLoader — Sequential (no shuffle, evaluate all)
    # ════════════════════════════════════════════════════════════
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,                 # Evaluate every single sample
        **common_kwargs,
    )

    logger.info(
        f"DataLoaders ready: "
        f"train={len(train_loader)} batches (bs={batch_size}), "
        f"val={len(val_loader)} batches"
    )

    return train_loader, val_loader
