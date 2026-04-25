import logging
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np

logger = logging.getLogger(__name__)

def build_dataloaders(train_dataset, val_dataset, cfg: dict):
    """
    Builds Dataloaders. 
    Implements WeightedRandomSampler if enabled in config.
    """
    dl_cfg             = cfg.get("dataloader", {})
    batch_size         = dl_cfg.get("batch_size", 4)
    num_workers        = dl_cfg.get("num_workers", 4)
    pin_memory         = dl_cfg.get("pin_memory", True)
    persistent_workers = dl_cfg.get("persistent_workers", True)
    prefetch_factor    = dl_cfg.get("prefetch_factor", 2)
    drop_last_train    = dl_cfg.get("drop_last", True)
    
    # Validation Loader is always sequential
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False
    )
    
    # Training Loader
    samp_cfg = cfg.get("sampling", {})
    if samp_cfg.get("enabled", True):
        from .sampler import TaskBalancedBatchSampler
        import os
        
        # Build Task-Balanced Batch Sampler
        train_sampler = TaskBalancedBatchSampler(
            task_indices=train_dataset.get_task_indices(),
            batch_size=batch_size,
            num_batches=samp_cfg.get("num_batches", 1000),
            rank=int(os.environ.get("RANK", "0")),
            world_size=int(os.environ.get("WORLD_SIZE", "1"))
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        logger.info(f"Using TaskBalancedBatchSampler: {batch_size} BS, 1000 batches/epoch")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=drop_last_train
        )
    
    return train_loader, val_loader
