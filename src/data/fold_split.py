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

    # [FIX] Khởi tạo RNG sớm để dùng cho sampling
    sampling_seed = config["split"].get("seed", 42)
    rng = random.Random(sampling_seed)

    # ─── PHÂN TÁCH LVO THÀNH 4 NHÓM CHIẾN THUẬT ──────────────────────────
    # [FIX] Theo yêu cầu: Cân bằng nội bộ LVO trước, sau đó mới nhân bản tổng thể
    lvo_df = df_train[df_train["has_lvo"] == 1]
    
    g_lvo_only = lvo_df[(lvo_df["has_lesion"] == 0) & (lvo_df["has_cow"] == 0)]["basename"].tolist()
    g_lvo_les  = lvo_df[(lvo_df["has_lesion"] == 1) & (lvo_df["has_cow"] == 0)]["basename"].tolist()
    g_lvo_cow  = lvo_df[(lvo_df["has_lesion"] == 0) & (lvo_df["has_cow"] == 1)]["basename"].tolist()
    g_lvo_all  = lvo_df[(lvo_df["has_lesion"] == 1) & (lvo_df["has_cow"] == 1)]["basename"].tolist()

    balanced_lvo_base = []
    # Nhóm 1: Duy nhất LVO (x3)
    balanced_lvo_base.extend(g_lvo_only * 3)
    # Nhóm 2: LVO + Lesion (x5)
    balanced_lvo_base.extend(g_lvo_les * 5)
    # Nhóm 3: LVO + CoW (x1.5) -> Nhân 1 lần và lấy thêm 50% ngẫu nhiên
    balanced_lvo_base.extend(g_lvo_cow)
    if len(g_lvo_cow) > 0:
        balanced_lvo_base.extend(rng.sample(g_lvo_cow, int(len(g_lvo_cow) * 0.5)))
    # Nhóm 4: Có cả 3 (x1)
    balanced_lvo_base.extend(g_lvo_all)

    # --- PHÂN TÁCH CÁC NHÓM CÒN LẠI ---
    other_list = df_train[(df_train["has_lvo"] == 0) & (df_train["has_lesion"] == 1)]["basename"].tolist()

    if "has_cow" in df_train.columns:
        neg_with_cow = df_train[(df_train["has_lvo"] == 0) & (df_train["has_lesion"] == 0) & (df_train["has_cow"] == 1)]["basename"].tolist()
        neg_plain    = df_train[(df_train["has_lvo"] == 0) & (df_train["has_lesion"] == 0) & (df_train["has_cow"] == 0)]["basename"].tolist()
    else:
        neg_with_cow, neg_plain = [], df_train[(df_train["has_lvo"] == 0) & (df_train["has_lesion"] == 0)]["basename"].tolist()

    sampled_basenames = []

    # 1. Thực hiện "Tổng tấn công LVO" (Nhân bản x6 tập đã cân bằng)
    final_lvo_factor = sampling_cfg.get("final_lvo_factor", 6)
    for _ in range(final_lvo_factor):
        sampled_basenames.extend(balanced_lvo_base)

    # 2. Giữ nguyên các ca có Lesion nhưng không LVO
    sampled_basenames.extend(other_list)

    # 3. Giữ 100% slice CoW(+) Lesion(-) — mỏ neo giải phẫu
    if cow_neg_keepall:
        sampled_basenames.extend(neg_with_cow)

    # 4. Downsample slice não trống
    n_plain = int(len(neg_plain) * plain_neg_ratio)
    if len(neg_plain) > 0:
        sampled_basenames.extend(rng.sample(neg_plain, min(n_plain, len(neg_plain))))

    path_map = {os.path.basename(f): f for f in train_files}
    final_list = [path_map[b] for b in sampled_basenames if b in path_map]
    rng.shuffle(final_list)

    print(f"\n[Sampling] CHIẾN THUẬT LVO CÂN BẰNG (Sub-total: {len(balanced_lvo_base)})")
    print(f"    - Duy nhất LVO: {len(g_lvo_only)} -> {len(g_lvo_only)*3}")
    print(f"    - LVO + Lesion: {len(g_lvo_les)} -> {len(g_lvo_les)*5}")
    print(f"    - LVO + CoW:    {len(g_lvo_cow)} -> {int(len(g_lvo_cow)*1.5)}")
    print(f"    - Có cả 3 nhãn: {len(g_lvo_all)} -> {len(g_lvo_all)}")
    print(f"    => TỔNG LVO (sau x{final_lvo_factor}): {len(balanced_lvo_base) * final_lvo_factor}")
    print(f"[Sampling] Lesion-only: {len(other_list)}")
    print(f"[Sampling] Background (plain) Downsampled: {len(neg_plain)} -> {n_plain}")
    print(f"[Sampling] Tổng số file Train cuối cùng: {len(final_list)}\n")

    return final_list

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
