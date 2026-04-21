"""
functional.py — Core metric functions for ISLES'24 3D evaluation.

All functions operate on numpy arrays (CPU) and are designed for
post-training evaluation only (too expensive for training loop).

Metrics:
  - dice_3d:             3D Volume Dice Score (Lesion, CoW)
  - recall_3d:           3D Volume Recall/Sensitivity (LVO)
  - hausdorff_95:        95th percentile Hausdorff Distance in mm (Lesion, CoW)
  - object_f1_centroid:  Object-level F1 with centroid radius detection (LVO)
"""
import logging
from typing import Dict, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def dice_3d(pred: np.ndarray, target: np.ndarray) -> float:
    """
    3D Volume Dice Similarity Coefficient.

    Dice = 2 * |pred ∩ target| / (|pred| + |target|)

    Used for: Lesion (infarct volume), CoW (vessel mapping)
    Clinical meaning: pixel-level overlap accuracy

    Args:
        pred:   [H, W, Z] binary uint8/float
        target: [H, W, Z] binary uint8/float

    Returns:
        Dice score in [0, 1]. Returns 1.0 if both are empty.
    """
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    smooth = 1e-6
    intersection = (pred * target).sum()
    cardinality = pred.sum() + target.sum()

    # Both empty → perfect agreement
    if cardinality < smooth:
        return 1.0

    return float((2.0 * intersection + smooth) / (cardinality + smooth))


def recall_3d(pred: np.ndarray, target: np.ndarray) -> float:
    """
    3D Volume Recall (Sensitivity).

    Recall = TP / (TP + FN) = |pred ∩ target| / |target|

    Used for: LVO detection
    Clinical meaning: "did we find the clot?" — missing a clot
    in emergency stroke is far worse than a false alarm.

    Args:
        pred:   [H, W, Z] binary
        target: [H, W, Z] binary

    Returns:
        Recall in [0, 1]. Returns 1.0 if target is empty (no clot to find).
    """
    target = target.astype(np.float64)
    pred = pred.astype(np.float64)

    target_sum = target.sum()
    if target_sum < 1e-6:
        return 1.0  # No clot → nothing to miss

    tp = (pred * target).sum()
    return float(tp / target_sum)


def hausdorff_95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> float:
    """
    95th percentile Hausdorff Distance in mm.

    Measures the maximum surface-to-surface distance (ignoring top 5% outliers).
    HD95 ↓ = more precise boundary delineation.

    Uses MONAI's compute_hausdorff_distance internally.

    Used for: Lesion, CoW
    Clinical meaning: how "artistic" is the AI's boundary tracing?
    Lower HD95 = closer to expert radiologist annotation.

    Args:
        pred:    [H, W, Z] binary
        target:  [H, W, Z] binary
        spacing: voxel spacing in mm (default 1mm isotropic for ISLES'24)

    Returns:
        HD95 distance in mm. Returns inf if either volume is empty.
    """
    pred_sum = pred.sum()
    target_sum = target.sum()

    # Edge cases
    if pred_sum == 0 and target_sum == 0:
        return 0.0
    if pred_sum == 0 or target_sum == 0:
        return float("inf")

    try:
        from monai.metrics import compute_hausdorff_distance

        # MONAI expects [B, C, H, W, D] float tensors
        pred_t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tgt_t = torch.from_numpy(target.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        hd = compute_hausdorff_distance(
            y_pred=pred_t,
            y=tgt_t,
            include_background=True,
            percentile=95,
            spacing=spacing,
        )
        return float(hd.item())
    except Exception as e:
        logger.warning(f"HD95 computation failed: {e}")
        return float("inf")


def object_f1_centroid(
    pred: np.ndarray,
    target: np.ndarray,
    radius: int = 3,
) -> Dict[str, float]:
    """
    Object-level F1 for LVO detection using centroid radius overlap.

    Algorithm:
      1. Find connected components in GT → compute centroids
      2. For each GT centroid: check if any pred pixel within radius r
         → TP (detected) or FN (missed)
      3. Find connected components in Pred
      4. For each Pred component: check if it overlaps any GT pixel
         → FP if no overlap (hallucinated clot)
      5. F1 = 2*TP / (2*TP + FP + FN)

    Uses radius=3 (relaxed) instead of exact pixel:
      - More forgiving of slight spatial offsets
      - Clinical reality: detecting the clot REGION is what matters
        for emergency treatment decisions

    Args:
        pred:   [H, W, Z] binary
        target: [H, W, Z] binary
        radius: pixel radius around centroid for detection (default=3)

    Returns:
        {
            "f1": float,   # Object-level F1 score
            "tp": int,     # Detected clots
            "fp": int,     # Hallucinated clots
            "fn": int,     # Missed clots
            "n_gt": int,   # Total GT objects
            "n_pred": int, # Total predicted objects
        }
    """
    from scipy.ndimage import label, center_of_mass

    # ── Find GT connected components ──
    gt_labels, n_gt = label(target.astype(np.int32))
    pred_labels, n_pred = label(pred.astype(np.int32))

    # Edge case: no GT and no predictions
    if n_gt == 0 and n_pred == 0:
        return {"f1": 1.0, "tp": 0, "fp": 0, "fn": 0, "n_gt": 0, "n_pred": 0}
    if n_gt == 0:
        return {"f1": 0.0, "tp": 0, "fp": n_pred, "fn": 0, "n_gt": 0, "n_pred": n_pred}
    if n_pred == 0:
        return {"f1": 0.0, "tp": 0, "fp": 0, "fn": n_gt, "n_gt": n_gt, "n_pred": 0}

    shape = pred.shape

    # ── Check GT centroids against predictions (with radius) ──
    tp = 0
    fn = 0

    for i in range(1, n_gt + 1):
        centroid = center_of_mass(target, gt_labels, i)

        # Build radius mask around centroid
        detected = False
        cx, cy, cz = [int(round(c)) for c in centroid]

        # Search within radius cube
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    # Check euclidean distance
                    if dx * dx + dy * dy + dz * dz > radius * radius:
                        continue

                    px = cx + dx
                    py = cy + dy
                    pz = cz + dz

                    # Bounds check
                    if (0 <= px < shape[0] and 0 <= py < shape[1]
                            and 0 <= pz < shape[2]):
                        if pred[px, py, pz] > 0:
                            detected = True
                            break
                if detected:
                    break
            if detected:
                break

        if detected:
            tp += 1
        else:
            fn += 1

    # ── Count FP: pred components not overlapping any GT ──
    fp = 0
    for j in range(1, n_pred + 1):
        pred_component = (pred_labels == j)
        if (pred_component & (target > 0)).sum() == 0:
            fp += 1

    # ── F1 ──
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)

    return {
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": n_gt,
        "n_pred": n_pred,
    }
