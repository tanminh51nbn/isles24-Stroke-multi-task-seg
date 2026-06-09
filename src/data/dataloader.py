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
import random
from torch.utils.data.dataloader import default_collate


class LesionCopyPasteCollate:
    """
    Lấy vùng Lesion từ sample A, paste vào sample B trong cùng batch.
    Tạo ra case tổng hợp hợp lệ về mặt giải phẫu.
    """
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, batch):
        # batch is a list of dicts {"input": tensor, "label": tensor, "basename": str}
        collated = default_collate(batch)
        
        inputs = collated["input"]   # (B, 18, H, W)
        labels = collated["label"]   # (B, 3, H, W)
        B = inputs.shape[0]
        
        for i in range(B):
            if random.random() > self.prob:
                continue
                
            # Tìm sample j có Lesion
            lesion_samples = [k for k in range(B) if labels[k, 0].sum() > 0]
            if not lesion_samples:
                continue
                
            j = random.choice(lesion_samples)
            lesion_mask = labels[j, 0:1]  # (1, H, W)
            
            if lesion_mask.sum() == 0:
                continue
                
            # Kiểm tra Perfusion compatibility (tránh paste lệch hệ)
            # Channel 6:12 chứa các thông tin CTP (CBV, CBF, Tmax, MTT, etc.)
            target_has_perf = (inputs[i, 6:12].abs().sum() > 1e-3)
            source_has_perf = (inputs[j, 6:12].abs().sum() > 1e-3)
            
            # Chỉ cho phép copy-paste nếu cả 2 cùng CÓ hoặc cùng KHÔNG CÓ CTP
            if target_has_perf != source_has_perf:
                continue
                
            # Paste Lesion từ j vào i
            inputs[i] = inputs[i] * (1 - lesion_mask) + inputs[j] * lesion_mask
            labels[i, 0] = torch.clamp(labels[i, 0] + labels[j, 0], 0, 1)
            
        collated["input"] = inputs
        collated["label"] = labels
        return collated


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
    aug_cfg = config.get("augmentation", {})
    
    collate_fn = None
    if aug_cfg.get("enabled", True) and "lesion_copy_paste" in aug_cfg:
        collate_fn = LesionCopyPasteCollate(prob=aug_cfg["lesion_copy_paste"]["prob"])

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
        collate_fn=collate_fn,
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
