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
        # Trả về 0.5 (hình phạt trung bình) cho các slice trống để ổn định gradient
        return np.ones_like(mask, dtype=np.float32) * 0.5
        
    dist_out = distance_transform_edt(1 - mask)
    dist_in  = distance_transform_edt(mask)
    
    # SDF = Khoảng cách ngoài - Khoảng cách trong
    # Giới hạn ở 20 pixel
    sdf = np.clip(dist_out - dist_in, -20, 20)
    
    # [QUY ĐỔI VỀ 0-1]: Chia cho 20 để đưa về thang đo [0, 1]
    # Lúc này: 0 là ranh giới, 1.0 là cực xa bên ngoài, -1.0 là cực sâu bên trong
    return (sdf / 20.0).astype(np.float32)


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
        inp = torch.nan_to_num(aug_inp, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Label gốc: float32, shape (3, 256, 256), values {0, 1}
        raw_label = torch.from_numpy(data["label"].astype(np.float32)).contiguous()

        # ── Augmentation ────────────────────────────────────────────────────
        sample = {"input": inp, "label": raw_label, "path": path}
        if self.transform is not None:
            sample = self.transform(sample)
        
        # Lấy lại data sau augment (đã là tensor)
        aug_inp = sample["input"]
        aug_lbl = sample["label"] # (3, 256, 256)

        # ── Lesion SDF Calculation (Hausdorff Guidance) ─────────────────────
        # Tính SDF sau khi Augment để đảm bảo khớp với mask đã xoay/biến dạng
        # Chuyển về numpy để dùng scipy bên trong compute_sdf
        lesion_mask_np = aug_lbl[0].cpu().numpy()
        lesion_sdf_np  = compute_sdf(lesion_mask_np)
        lesion_sdf_ts  = torch.from_numpy(lesion_sdf_np).to(aug_lbl.device)

        # ── Final Label Assembly ────────────────────────────────────────────
        # Gộp thành label 4 kênh: [Lesion_Mask, LVO_Binary, CoW_Mask, Lesion_SDF]
        # Kênh 1 (LVO) lúc này đã là binary vì chúng ta đã bỏ bước tạo heatmap ở dataset
        full_label = torch.cat([
            aug_lbl, # [Lesion, LVO, CoW]
            lesion_sdf_ts.unsqueeze(0) # [SDF]
        ], dim=0)

        # Sanitize NaN/inf (Cuối cùng cho an toàn tuyệt đối)
        lbl = torch.nan_to_num(full_label, nan=0.0, posinf=0.0, neginf=0.0)

        return {"input": inp, "label": lbl, "path": path}




def build_dataset(
    file_list: List[str],
    transform: Optional[Callable] = None,
) -> ISLES24Dataset:
    return ISLES24Dataset(file_list, transform)
