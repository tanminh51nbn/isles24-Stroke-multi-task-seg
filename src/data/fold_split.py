"""
fold_split.py — Scan 2 Kaggle dataset parts, compute lesion volumes,
                perform Stratified K-Fold at patient level.

Usage:
    splits, patient_dirs = build_kfold_splits(
        data_dirs=["/kaggle/input/part1", "/kaggle/input/part2"],
        cfg=yaml_cfg["kfold"],
        cache_dir="/kaggle/working",
    )
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Internal helpers
# ═════════════════════════════════════════════════════════════════

def _discover_patients(data_dirs: List[str]) -> Dict[str, Path]:
    """
    Scan multiple dataset directories, merge into a single patient registry.

    Returns:
        Dict mapping patient_id (e.g. "sub-stroke0001") → absolute directory Path
    """
    patients: Dict[str, Path] = {}
    for d in data_dirs:
        root = Path(d)
        if not root.exists():
            logger.warning(f"Data directory does not exist: {root}")
            continue
        for patient_dir in sorted(root.iterdir()):
            if patient_dir.is_dir() and patient_dir.name.startswith("sub-stroke"):
                pid = patient_dir.name
                if pid in patients:
                    logger.warning(
                        f"Duplicate patient {pid} found in {root}, "
                        f"keeping first occurrence from {patients[pid].parent}"
                    )
                    continue
                patients[pid] = patient_dir

    logger.info(
        f"Discovered {len(patients)} patients from {len(data_dirs)} directories."
    )
    return patients


def _compute_lesion_volumes(
    patient_dirs: Dict[str, Path],
    cache_path: Path,
) -> Dict[str, int]:
    """
    Compute total lesion pixel count (channel 0) for each patient.
    Results are cached to a JSON file for instant reload on subsequent runs.

    First run: scans ~149 patients × ~80 slices ≈ 10 GB I/O (~2-3 min on NVMe).
    Subsequent runs: loads JSON in < 0.1s.
    """
    # ── Try cache first ──
    if cache_path.exists():
        logger.info(f"Loading cached lesion volumes from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    # ── Full scan ──
    logger.info(
        "Computing lesion volumes for stratification "
        "(first run only, scanning all label files)..."
    )
    volumes: Dict[str, int] = {}

    for i, (pid, pdir) in enumerate(sorted(patient_dirs.items())):
        label_dir = pdir / "labels"
        total_pixels = 0
        for lbl_file in sorted(label_dir.glob("y_z*.npy")):
            y = np.load(lbl_file)           # [3, 544, 544] uint8
            total_pixels += int(y[0].sum())  # Channel 0 = lesion
        volumes[pid] = total_pixels

        if (i + 1) % 25 == 0:
            logger.info(f"  Scanned {i + 1}/{len(patient_dirs)} patients...")

    # ── Persist cache ──
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(volumes, f, indent=2)
    logger.info(f"Cached lesion volumes → {cache_path}")

    return volumes


# ═════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════

def build_kfold_splits(
    data_dirs: List[str],
    cfg: dict,
    cache_dir: str = ".",
) -> Tuple[List[Tuple[List[str], List[str]]], Dict[str, Path]]:
    """
    Build Stratified K-Fold splits at patient level.

    Stratification is based on quantile-binned lesion volume, ensuring
    each fold has a representative distribution of lesion severity.

    Args:
        data_dirs: Paths to Kaggle dataset parts
                   (e.g. ["/kaggle/input/part1", "/kaggle/input/part2"])
        cfg:       kfold section from data.yaml
        cache_dir: Directory to store lesion_volumes.json cache

    Returns:
        splits:       List of (train_patient_ids, val_patient_ids) per fold
        patient_dirs: Dict mapping patient_id → absolute directory Path
    """
    # 1. Discover all patients across both parts
    patient_dirs = _discover_patients(data_dirs)
    patient_ids = sorted(patient_dirs.keys())
    n_patients = len(patient_ids)

    if n_patients == 0:
        raise ValueError(
            f"No patients found in data_dirs: {data_dirs}. "
            "Ensure directories contain sub-strokeXXXX folders."
        )

    # 2. Compute lesion volumes (with JSON caching)
    cache_path = Path(cache_dir) / "lesion_volumes.json"
    volumes = _compute_lesion_volumes(patient_dirs, cache_path)

    # 3. Quantile binning for stratification
    n_bins = cfg.get("n_bins", 5)
    vol_array = np.array([volumes.get(pid, 0) for pid in patient_ids])

    # Separate zero-volume patients (no lesion at all) into bin 0
    nonzero_vols = vol_array[vol_array > 0]
    if len(nonzero_vols) > 0:
        bin_edges = np.quantile(
            nonzero_vols,
            np.linspace(0, 1, n_bins + 1)[1:]  # Exclude 0th percentile
        )
        bins = np.digitize(vol_array, bin_edges, right=True)
    else:
        bins = np.zeros(n_patients, dtype=int)

    # Zero-volume patients → explicit bin 0
    bins[vol_array == 0] = 0

    logger.info(
        f"Stratification bins: {dict(zip(*np.unique(bins, return_counts=True)))}"
    )

    # 4. Stratified K-Fold
    n_splits = cfg.get("n_splits", 5)
    random_state = cfg.get("random_state", 42)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, bins)):
        train_ids = [patient_ids[i] for i in train_idx]
        val_ids = [patient_ids[i] for i in val_idx]
        logger.info(
            f"Fold {fold_idx}: "
            f"train={len(train_ids)} patients, "
            f"val={len(val_ids)} patients"
        )
        splits.append((train_ids, val_ids))

    return splits, patient_dirs
