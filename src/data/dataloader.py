"""
dataloader.py — Xây dựng DataLoader cho training phân tán (DDP 2 GPU)

Chiến lược:
    - Train: DistributedSampler (shuffle=True) → đảm bảo mỗi GPU thấy dữ liệu khác nhau
    - Val:   DistributedSampler (shuffle=False) → đánh giá nhất quán
    - pin_memory=True: Tăng tốc transfer CPU→GPU
    - persistent_workers=True: Tránh tạo lại worker process mỗi epoch
"""

import torch
from torch.utils.data import DataLoader, DistributedSampler
from typing import Tuple

from .dataset import ISLES24Dataset
from .transforms import build_train_transforms, build_val_transforms
from .fold_split import build_patient_split, apply_sampling, build_stratified_kfold_splits

import os
import glob


def worker_init_fn(worker_id):
    import torch
    torch.set_num_threads(1)


def build_dataloaders(
    config: dict,
    dataset_dir: str,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Xây dựng train/val DataLoader từ config và thư mục dataset.

    Args:
        config:       Dict đọc từ data.yaml
        dataset_dir:  Đường dẫn đến thư mục chứa file .npy
        rank:         GPU rank hiện tại (DDP)
        world_size:   Tổng số GPU

    Returns:
        (train_loader, val_loader)
    """
    # Thu thập tất cả file
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.npy")))
    if len(all_files) == 0:
        raise FileNotFoundError(f"Không tìm thấy file .npy trong: {dataset_dir}")

    # Chia train/val theo bệnh nhân
    split_cfg  = config["split"]
    split_mode = split_cfg.get("mode", "single")

    if split_mode == "kfold":
        # [FIX 3] Stratified K-Fold: lấy đúng fold theo current_fold
        metadata_csv = config["sampling"]["metadata_csv"]
        n_folds      = split_cfg.get("n_folds", 5)
        current_fold = split_cfg.get("current_fold", 0)

        all_splits = build_stratified_kfold_splits(
            file_list=all_files,
            metadata_csv=metadata_csv,
            n_folds=n_folds,
            seed=split_cfg.get("seed", 42),
        )
        train_files, val_files = all_splits[current_fold]

        if rank == 0:
            print(f"[DataLoader] K-Fold mode: Fold {current_fold + 1}/{n_folds}")
    else:
        # Single split (chế độ cũ)
        train_files, val_files = build_patient_split(
            all_files,
            val_ratio=split_cfg["val_ratio"],
            seed=split_cfg["seed"],
        )

    # Lưu lại danh sách gốc trước khi sampling để dùng cho Cyclic Stride
    train_files_original = list(train_files)
    
    # Thực hiện Smart Sampling (khởi tạo cho Epoch 0)
    train_files = apply_sampling(train_files, config, epoch=0)

    # Build Dataset
    train_dataset = ISLES24Dataset(train_files, transform=build_train_transforms(config))
    val_dataset   = ISLES24Dataset(val_files,   transform=build_val_transforms())

    # Sampler cho DDP
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None

    dl_cfg = config["dataloader"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=dl_cfg["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        persistent_workers=dl_cfg.get("persistent_workers", True),
        prefetch_factor=dl_cfg.get("prefetch_factor", 2),
        drop_last=dl_cfg.get("drop_last", True),
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=dl_cfg["batch_size"],
        sampler=val_sampler,
        shuffle=False,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        persistent_workers=dl_cfg.get("persistent_workers", True),
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    if rank == 0:
        print(f"[DataLoader] Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    return train_loader, val_loader, train_files_original
