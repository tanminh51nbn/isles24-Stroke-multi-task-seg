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


def apply_sampling(
    train_files: List[str],
    downsample_neg_ratio: float = 0.3,
    lvo_oversample_factor: int = 5,
) -> List[str]:
    """
    Cân bằng tập train bằng cách:
        - Giữ lại một phần slice all-negative (background)
        - Lặp lại slice có LVO nhiều lần (oversample)

    Lưu ý: Hàm này cần metadata để biết file nào có LVO.
    Nếu không có metadata, dùng tên file để đoán (không chính xác).
    Tốt hơn là nạp từ dataset_metadata.csv.

    Args:
        train_files:           Danh sách file train
        downsample_neg_ratio:  Tỷ lệ giữ lại slice all-negative
        lvo_oversample_factor: Số lần lặp slice có LVO

    Returns:
        Danh sách file đã được cân bằng (có thể có duplicate)
    """
    # Placeholder — logic đầy đủ cần đọc metadata CSV
    # PINELINE.py sẽ truyền metadata vào để thực hiện sampling chính xác
    return train_files
