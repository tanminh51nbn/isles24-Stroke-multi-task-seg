from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
from sklearn.model_selection import train_test_split

def build_kfold_splits(
    data_dir: Union[str, List[str]],
    cfg: dict
) -> Tuple[List[Path], List[Path]]:
    """
    Splits patient directories into train/val.
    - Hỗ trợ single path hoặc list of paths (multi-root cho Part_1 + Part_2)
    - Stratified theo (has_lvo, has_lesion) để đảm bảo val set đại diện
    """
    # Lấy tham số từ config
    k_cfg = cfg.get("kfold", {})
    val_ratio = k_cfg.get("val_ratio", 0.2)
    seed = k_cfg.get("seed", 42)

    # Thu thập tất cả patient dirs từ một hoặc nhiều root
    if isinstance(data_dir, (list, tuple)):
        patients = []
        for d in data_dir:
            patients.extend(
                sorted([p for p in Path(d).iterdir() if p.is_dir()])
            )
    else:
        patients = sorted([p for p in Path(data_dir).iterdir() if p.is_dir()])

    if len(patients) == 0:
        raise ValueError(f"No patient directories found in {data_dir}")

    # Tạo stratify label dựa trên has_lvo + has_lesion
    # Check 5 slice quanh vùng giữa (Z median=0.55) vì LVO tập trung ở đó
    strat_labels = []
    for p in patients:
        label_files = sorted((p / "labels").glob("y_z*.npy"))
        has_lvo = False
        has_lesion = False
        
        mid = len(label_files) // 2
        sample_files = label_files[max(0, mid - 2): mid + 3]  # 5 slice quanh vùng giữa
        for lf in sample_files:
            lbl = np.load(lf, mmap_mode='r')
            if lbl[1].max() > 0:
                has_lvo = True
            if lbl[0].max() > 0:
                has_lesion = True
            if has_lvo and has_lesion:
                break

        if has_lvo and has_lesion:
            strat_labels.append("lvo_les")
        elif has_lesion:
            strat_labels.append("les_only")
        elif has_lvo:
            strat_labels.append("lvo_only")
        else:
            strat_labels.append("neg")

    train_patients, val_patients = train_test_split(
        patients,
        test_size=val_ratio,
        random_state=seed,
        stratify=strat_labels
    )

    return train_patients, val_patients
