"""
src.metrics — ISLES'24 Medical-grade Metrics

Post-training evaluation module for 3D volume-level metrics.
(2D validation metrics are computed inline in src.engine.trainer)

Public API:
    Functional (core metric functions):
        dice_3d              → 3D Volume Dice Score
        recall_3d            → 3D Volume Recall/Sensitivity
        hausdorff_95         → 95th percentile Hausdorff Distance (mm)
        object_f1_centroid   → Object-level F1 with centroid detection

    Evaluator (full pipeline):
        Evaluator            → Predict + reconstruct 3D + compute all metrics
"""
from .evaluator import Evaluator
from .functional import dice_3d, hausdorff_95, object_f1_centroid, recall_3d

__all__ = [
    # Functional
    "dice_3d",
    "recall_3d",
    "hausdorff_95",
    "object_f1_centroid",
    # Pipeline
    "Evaluator",
]
