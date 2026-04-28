"""
dataset.py — PyTorch Dataset cho ISLES'24 NPY files

Mỗi file .npy là một dict:
    'input': float32 (18, 256, 256) — 18-channel 2.5D CT
    'label': uint8   (3,  256, 256) — 3 binary masks (Lesion, LVO, CoW)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Callable


class ISLES24Dataset(Dataset):
    """
    Dataset nạp các file .npy đã được tiền xử lý.

    Args:
        file_list: Danh sách đường dẫn tuyệt đối đến từng file .npy
        transform:  Optional transform áp dụng lên cặp (input, label)
    """

    def __init__(self, file_list: List[str], transform: Optional[Callable] = None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        path = self.file_list[idx]
        data = np.load(path, allow_pickle=True).item()

        # Input: float32, shape (18, 256, 256), range [0, 1]
        inp = torch.from_numpy(data["input"].astype(np.float32))

        # Label: float32 (BCEWithLogitsLoss cần float), shape (3, 256, 256), values {0, 1}
        lbl = torch.from_numpy(data["label"].astype(np.float32))

        sample = {"input": inp, "label": lbl, "path": path}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def build_dataset(
    file_list: List[str],
    transform: Optional[Callable] = None,
) -> ISLES24Dataset:
    return ISLES24Dataset(file_list, transform)
