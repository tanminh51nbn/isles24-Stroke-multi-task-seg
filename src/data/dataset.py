"""
dataset.py — PyTorch Dataset cho ISLES'24 NPY files

Mỗi file .npy là một dict:
    'input': float32 (18, 256, 256) — 18-channel 2.5D CT
    'label': uint8   (3,  256, 256) — 3 binary masks (Lesion, LVO, CoW)

Lưu ý về LVO Label:
    Nhãn LVO (label[1]) được biến đổi thành Gaussian Heatmap on-the-fly.
    Thay vì mask nhị phân {0, 1}, heatmap LVO có giá trị liên tục [0, 1]
    với đỉnh 1.0 tại tâm tổn thương, giảm dần theo đường cong Gaussian ra ngoài.
    Kỹ thuật này giải quyết class imbalance cực đoan và cung cấp gradient phong
    phú hơn cho bài toán phát hiện điểm nhỏ (keypoint-style detection).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Callable
from scipy.ndimage import gaussian_filter, distance_transform_edt


# ─── Gaussian Heatmap Generator ───────────────────────────────────────────────

def make_lvo_heatmap(binary_mask: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """
    Chuyển đổi mask LVO nhị phân thành Gaussian Heatmap.

    Cơ chế:
        1. Làm mờ mask nhị phân bằng bộ lọc Gaussian (sigma pixel).
        2. Normalize sao cho đỉnh của vùng sáng = 1.0.
        3. Nếu mask rỗng (không có LVO), trả về ma trận 0 nguyên xi.

    Args:
        binary_mask: np.ndarray, shape (H, W), dtype float, values {0, 1}
        sigma:       Độ rộng của quầng sáng Gaussian (pixel). Lớn hơn = lan rộng hơn.

    Returns:
        heatmap: np.ndarray, shape (H, W), dtype float32, values [0, 1]
    """
    if binary_mask.sum() == 0:
        return binary_mask.astype(np.float32)  # Không có LVO → trả về toàn 0

    heatmap = gaussian_filter(binary_mask.astype(np.float32), sigma=sigma)

    # Normalize: đỉnh phải là 1.0
    peak = heatmap.max()
    if peak > 0:
        heatmap = heatmap / peak

    return heatmap.astype(np.float32)

def compute_sdf(mask: np.ndarray) -> np.ndarray:
    """
    Tính toán Signed Distance Function (SDF) từ mask nhị phân.
    - Ngoài vật thể: Giá trị dương (khoảng cách tới biên gần nhất).
    - Trong vật thể: Giá trị âm (khoảng cách tới biên gần nhất).
    """
    if mask.sum() == 0:
        # Nếu không có lesion, SDF là một hằng số lớn (mô phỏng vô tận)
        return np.ones_like(mask, dtype=np.float32) * 255.0
        
    dist_out = distance_transform_edt(1 - mask)
    dist_in  = distance_transform_edt(mask)
    
    # SDF = Khoảng cách ngoài - Khoảng cách trong
    sdf = dist_out - dist_in
    return sdf.astype(np.float32)


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

        # Label gốc: float32, shape (3, 256, 256), values {0, 1}
        raw_label = data["label"].astype(np.float32)

        # ── LVO Heatmap Conversion ───────────────────────────────────────────
        lvo_heatmap = make_lvo_heatmap(raw_label[1], sigma=4.0)
        raw_label[1] = lvo_heatmap

        # ── Lesion SDF Calculation (Hausdorff Guidance) ─────────────────────
        lesion_sdf = compute_sdf(raw_label[0])
        
        # Gộp thành label 4 kênh: [Lesion_Mask, LVO_Heatmap, CoW_Mask, Lesion_SDF]
        full_label = np.concatenate([raw_label, lesion_sdf[None, ...]], axis=0)
        lbl = torch.from_numpy(full_label)
        # ────────────────────────────────────────────────────────────────────

        # ── Sanitize NaN/inf ────────────────────────────────────────────────
        # Một số kênh Perfusion (Tmax, CBF, CBV, MTT) có thể chứa NaN/inf
        # do thiếu dữ liệu scan hoặc lỗi trong bước tiền xử lý (chia cho 0).
        # Thay NaN → 0.0, +inf → 0.0, -inf → 0.0 để giữ tensor hợp lệ.
        # nan_to_num an toàn: không thay đổi giá trị bình thường, chỉ xử lý giá trị lỗi.
        inp = torch.nan_to_num(inp, nan=0.0, posinf=0.0, neginf=0.0)
        lbl = torch.nan_to_num(lbl, nan=0.0, posinf=0.0, neginf=0.0)
        # ────────────────────────────────────────────────────────────────────

        sample = {"input": inp, "label": lbl, "path": path}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample



def build_dataset(
    file_list: List[str],
    transform: Optional[Callable] = None,
) -> ISLES24Dataset:
    return ISLES24Dataset(file_list, transform)
