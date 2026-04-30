"""
fold_split.py — Chia train/val theo patient-level (GroupShuffleSplit)

Nguyên tắc bắt buộc:
    TOÀN BỘ lát cắt của một bệnh nhân phải nằm trong CÙNG một tập.
    Nếu vi phạm → Data Leakage → mô hình "học bài" → điểm validation ảo.

Cách hoạt động:
    1. Trích patient_id từ tên file (sub-stroke0001_slice030.npy → 0001)
    2. Chia 149 bệnh nhân theo tỷ lệ (80/20)
    3. Map bệnh nhân → danh sách file tương ứng
"""

import os
import random
from typing import List, Tuple


def _extract_patient_id(filename: str) -> str:
    """
    Trích patient ID từ tên file.
    'sub-stroke0001_slice030.npy' → 'sub-stroke0001'
    """
    basename = os.path.basename(filename)
    return basename.split("_")[0]


def build_patient_split(
    file_list: List[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Chia file_list thành train/val theo patient-level GroupShuffleSplit.

    Args:
        file_list: Danh sách đường dẫn tất cả file .npy
        val_ratio: Tỷ lệ bệnh nhân dùng cho validation (default 0.2 = 20%)
        seed:      Random seed để tái lập kết quả

    Returns:
        (train_files, val_files): 2 danh sách file, không overlap bệnh nhân
    """
    # Nhóm file theo bệnh nhân
    patient_to_files: dict = {}
    for f in file_list:
        pid = _extract_patient_id(f)
        patient_to_files.setdefault(pid, []).append(f)

    # Shuffle danh sách bệnh nhân
    patients = sorted(patient_to_files.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    # Chia bệnh nhân
    n_val = max(1, int(len(patients) * val_ratio))
    val_patients  = set(patients[:n_val])
    train_patients = set(patients[n_val:])

    # Map bệnh nhân → file
    train_files = [f for pid in train_patients for f in patient_to_files[pid]]
    val_files   = [f for pid in val_patients   for f in patient_to_files[pid]]

    print(f"[fold_split] Bệnh nhân Train: {len(train_patients)} | Val: {len(val_patients)}")
    print(f"[fold_split] Slice Train: {len(train_files)} | Val: {len(val_files)}")

    return train_files, val_files


import pandas as pd

def apply_sampling(
    train_files: List[str],
    config: dict,
) -> List[str]:
    """
    Cân bằng tập train dựa trên cấu hình trong data.yaml.
    """
    sampling_cfg = config["sampling"]
    if not sampling_cfg.get("enabled", False):
        return train_files

    metadata_csv = sampling_cfg["metadata_csv"]
    if not os.path.exists(metadata_csv):
        print(f"[Warning] Không tìm thấy {metadata_csv}. Bỏ qua sampling.")
        return train_files

    lvo_oversample_factor = sampling_cfg.get("lvo_oversample_factor", 5)
    downsample_neg_ratio  = sampling_cfg.get("downsample_neg_ratio", 0.3)

    df = pd.read_csv(metadata_csv)
    # Chuẩn hóa đường dẫn trong CSV về dạng basename để so sánh environment-agnostic
    df["basename"] = df["path"].apply(os.path.basename)

    # Lọc chỉ lấy những file nằm trong tập train_files hiện tại
    train_basenames = {os.path.basename(f) for f in train_files}
    df_train = df[df["basename"].isin(train_basenames)].copy()

    # Phân loại (Sử dụng basename để đồng bộ)
    lvo_list = df_train[df_train["has_lvo"] == 1]["basename"].tolist()
    neg_list = df_train[(df_train["has_lvo"] == 0) & (df_train["has_lesion"] == 0)]["basename"].tolist()
    other_list = df_train[(df_train["has_lvo"] == 0) & (df_train["has_lesion"] == 1)]["basename"].tolist()

    # Thực hiện lấy mẫu
    sampled_basenames = []
    
    # 1. Nhân bản LVO
    for _ in range(lvo_oversample_factor):
        sampled_basenames.extend(lvo_list)
    
    # 2. Giữ nguyên các ca có Lesion nhưng không LVO
    sampled_basenames.extend(other_list)

    # 3. Downsample các ca toàn màu đen
    n_neg = int(len(neg_list) * downsample_neg_ratio)
    if len(neg_list) > 0:
        sampled_basenames.extend(random.sample(neg_list, n_neg))

    # Chuyển basename ngược lại thành full path của môi trường hiện tại
    path_map = {os.path.basename(f): f for f in train_files}
    final_list = [path_map[b] for b in sampled_basenames]
    
    random.shuffle(final_list)
    
    print(f"[Sampling] LVO: {len(lvo_list)} -> {len(lvo_list)*lvo_oversample_factor}")
    print(f"[Sampling] Background Downsampled: {len(neg_list)} -> {n_neg}")
    print(f"[Sampling] Tổng số file sau khi balance: {len(final_list)}")

    return final_list
