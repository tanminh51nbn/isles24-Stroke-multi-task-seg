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
    
    # [FIX 1] Thêm các config mới cho counter-example sampling
    cow_neg_keepall = sampling_cfg.get("cow_neg_keepall", True)
    plain_neg_ratio = sampling_cfg.get("plain_neg_ratio", downsample_neg_ratio)

    df = pd.read_csv(metadata_csv)
    df["basename"] = df["path"].apply(os.path.basename)

    train_basenames = {os.path.basename(f) for f in train_files}
    df_train = df[df["basename"].isin(train_basenames)].copy()

    # Phân loại
    lvo_list   = df_train[df_train["has_lvo"] == 1]["basename"].tolist()
    other_list = df_train[(df_train["has_lvo"] == 0) & (df_train["has_lesion"] == 1)]["basename"].tolist()

    # [FIX 1] Tách neg_list thành 2 nhóm theo has_cow
    # Nhóm vàng: CoW(+), Lesion(-) → Đây là counter-examples dạy AI "mạch máu lành ≠ tổn thương"
    # Được giữ 100% không cắt xén, bất kể downsample_neg_ratio là bao nhiêu.
    if "has_cow" in df_train.columns:
        neg_with_cow = df_train[
            (df_train["has_lvo"] == 0) &
            (df_train["has_lesion"] == 0) &
            (df_train["has_cow"] == 1)
        ]["basename"].tolist()
        neg_plain = df_train[
            (df_train["has_lvo"] == 0) &
            (df_train["has_lesion"] == 0) &
            (df_train["has_cow"] == 0)
        ]["basename"].tolist()
    else:
        # Fallback nếu metadata không có cột has_cow
        neg_with_cow = []
        neg_plain = df_train[
            (df_train["has_lvo"] == 0) & (df_train["has_lesion"] == 0)
        ]["basename"].tolist()

    sampling_seed = config["split"].get("seed", 42)
    rng = random.Random(sampling_seed)

    sampled_basenames = []

    # 1. Nhân bản LVO
    for _ in range(lvo_oversample_factor):
        sampled_basenames.extend(lvo_list)

    # 2. Giữ nguyên các ca có Lesion nhưng không LVO
    sampled_basenames.extend(other_list)

    # 3. [FIX 1] Giữ 100% slice CoW(+) Lesion(-) — counter-examples
    if cow_neg_keepall and neg_with_cow:
        sampled_basenames.extend(neg_with_cow)
        print(f"[Sampling] CoW+ Lesion- (counter-examples): {len(neg_with_cow)} (GIỮ 100%)")

    # 4. Downsample slice não trống hoàn toàn (không CoW, không Lesion, không LVO)
    n_plain = int(len(neg_plain) * plain_neg_ratio)
    if len(neg_plain) > 0:
        sampled_basenames.extend(rng.sample(neg_plain, min(n_plain, len(neg_plain))))

    path_map = {os.path.basename(f): f for f in train_files}
    final_list = [path_map[b] for b in sampled_basenames if b in path_map]

    rng.shuffle(final_list)

    print(f"[Sampling] LVO: {len(lvo_list)} -> {len(lvo_list)*lvo_oversample_factor}")
    print(f"[Sampling] Lesion-only: {len(other_list)}")
    print(f"[Sampling] Background (plain) Downsampled: {len(neg_plain)} -> {n_plain}")
    print(f"[Sampling] Tổng số file sau khi balance: {len(final_list)}")

    return final_list


def build_stratified_kfold_splits(
    file_list: List[str],
    metadata_csv: str,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """
    [FIX 3] Stratified K-Fold theo LVO status bệnh nhân.

    Đảm bảo mỗi fold có tỷ lệ bệnh nhân LVO đồng đều, tránh việc
    một fold vô tình không có LVO patient nào trong val set.

    Args:
        file_list:    Tất cả file .npy của tập train+val
        metadata_csv: Đường dẫn tới metadata CSV
        n_folds:      Số fold (mặc định 5)
        seed:         Random seed

    Returns:
        List n_folds cặp (train_files, val_files), không overlap bệnh nhân
    """
    import pandas as pd

    df = pd.read_csv(metadata_csv)
    df["basename"] = df["path"].apply(os.path.basename)

    # Nhóm file theo bệnh nhân
    patient_to_files: dict = {}
    for f in file_list:
        pid = _extract_patient_id(f)
        patient_to_files.setdefault(pid, []).append(f)

    # Xác định LVO status của từng bệnh nhân
    # (bệnh nhân có LVO nếu BấT KỂ slice nào có has_lvo == 1)
    patient_lvo_status = {}
    for pid in patient_to_files:
        basenames = {os.path.basename(f) for f in patient_to_files[pid]}
        patient_df = df[df["basename"].isin(basenames)]
        has_lvo = (patient_df["has_lvo"] == 1).any() if "has_lvo" in patient_df.columns else False
        patient_lvo_status[pid] = 1 if has_lvo else 0

    # Tách 2 nhóm bệnh nhân
    rng = random.Random(seed)
    lvo_patients     = sorted([p for p, v in patient_lvo_status.items() if v == 1])
    non_lvo_patients = sorted([p for p, v in patient_lvo_status.items() if v == 0])
    rng.shuffle(lvo_patients)
    rng.shuffle(non_lvo_patients)

    # Tạo K-Fold đồng đều cho từng nhóm
    def chunk(lst, k):
        """Chia lst thành k phần gần bằng nhau."""
        n = len(lst)
        return [lst[i * n // k:(i + 1) * n // k] for i in range(k)]

    lvo_folds     = chunk(lvo_patients, n_folds)
    non_lvo_folds = chunk(non_lvo_patients, n_folds)

    splits = []
    for fold_idx in range(n_folds):
        val_patients  = set(lvo_folds[fold_idx] + non_lvo_folds[fold_idx])
        train_patients = set(patient_to_files.keys()) - val_patients

        train_files = [f for pid in train_patients for f in patient_to_files[pid]]
        val_files   = [f for pid in val_patients   for f in patient_to_files[pid]]

        n_lvo_val = sum(patient_lvo_status[p] for p in val_patients)
        print(f"[KFold] Fold {fold_idx}: Train={len(train_patients)} patients, "
              f"Val={len(val_patients)} patients ({n_lvo_val} LVO)")

        splits.append((train_files, val_files))

    return splits
