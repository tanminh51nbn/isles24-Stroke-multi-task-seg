"""
evaluator.py — Post-training 3D evaluation pipeline for ISLES'24.

Reconstructs 3D volumes from 2D slice predictions, then computes
medical-grade metrics per patient. Designed to run AFTER training completes.

Workflow:
  1. Load best checkpoint
  2. For each patient: predict all 2D slices → stack into 3D [H, W, Z]
  3. Compute: 3D Dice (Lesion, CoW), HD95 (Lesion, CoW),
              Recall (LVO), Object F1 (LVO centroid r=3)
  4. Aggregate: mean ± std across patients
  5. Print formatted clinical report
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .functional import dice_3d, hausdorff_95, object_f1_centroid, recall_3d

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Post-training 3D evaluation for ISLES'24 Multi-Task Segmentation.

    Computes medical-grade metrics on full 3D brain volumes,
    separate from the 2D metrics used during training validation.
    """

    TASK_NAMES = ("lesion", "lvo", "cow")

    def __init__(
        self,
        model: nn.Module,
        patient_ids: List[str],
        patient_dirs: Dict[str, Path],
        device: str = "cuda",
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        centroid_radius: int = 3,
    ):
        """
        Args:
            model:           Trained MultiTaskUNet (already loaded with weights)
            patient_ids:     List of patient IDs to evaluate
            patient_dirs:    {patient_id: Path} mapping from fold_split
            device:          "cuda" or "cpu"
            spacing:         Voxel spacing in mm (default 1mm isotropic for ISLES'24)
            centroid_radius: Radius for LVO centroid detection (default 3 pixels)
        """
        self.model = model.to(device)
        self.model.eval()
        self.patient_ids = patient_ids
        self.patient_dirs = patient_dirs
        self.device = device
        self.spacing = spacing
        self.centroid_radius = centroid_radius

    def evaluate_all(self) -> Dict:
        """
        Run full 3D evaluation on all patients.

        Returns:
            results: {
              "per_patient": {pid: {task: {metric: value}}},
              "summary": {
                "lesion_dice_3d_mean": float, "lesion_dice_3d_std": float,
                "lesion_hd95_mean": float,    "lesion_hd95_std": float,
                "lvo_recall_3d_mean": float,  "lvo_recall_3d_std": float,
                "lvo_object_f1_mean": float,  "lvo_object_f1_std": float,
                "cow_dice_3d_mean": float,    "cow_dice_3d_std": float,
                "cow_hd95_mean": float,       "cow_hd95_std": float,
              }
            }
        """
        per_patient = {}
        n_total = len(self.patient_ids)

        logger.info(f"Evaluating {n_total} patients (3D volume metrics)...")

        for idx, pid in enumerate(self.patient_ids):
            logger.info(f"  [{idx+1}/{n_total}] Patient {pid}")

            try:
                pred_3d, gt_3d = self._predict_patient_3d(pid)
                patient_metrics = self._compute_patient_metrics(pred_3d, gt_3d)
                per_patient[pid] = patient_metrics
            except Exception as e:
                logger.error(f"  ❌ Patient {pid} failed: {e}")
                per_patient[pid] = None

        # ── Aggregate ──
        summary = self._aggregate(per_patient)

        return {
            "per_patient": per_patient,
            "summary": summary,
        }

    @torch.no_grad()
    def _predict_patient_3d(
        self, pid: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Predict all 2D slices for one patient, stack into 3D volumes.

        Args:
            pid: Patient ID

        Returns:
            pred_3d: list of 3 arrays, each [H, W, Z] binary (uint8)
            gt_3d:   list of 3 arrays, each [H, W, Z] binary (uint8)
        """
        patient_dir = Path(self.patient_dirs[pid])

        # ── Find all slice files (sorted by Z index) ──
        x_files = sorted(patient_dir.glob("x_z*.npy"))
        y_files = sorted(patient_dir.glob("y_z*.npy"))

        if len(x_files) == 0:
            raise FileNotFoundError(f"No slice files in {patient_dir}")

        if len(x_files) != len(y_files):
            raise ValueError(
                f"Mismatch: {len(x_files)} x-files vs {len(y_files)} y-files "
                f"in {patient_dir}"
            )

        pred_slices = [[] for _ in range(3)]  # 3 tasks
        gt_slices = [[] for _ in range(3)]

        for x_path, y_path in zip(x_files, y_files):
            # Load
            x = np.load(x_path).astype(np.float32)  # [18, H, W]
            y = np.load(y_path)  # [3, H, W]

            # Predict
            x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                preds = self.model(x_tensor)  # list of 3 × [1, 1, H, W]

            # Threshold + collect
            for i in range(3):
                pred_2d = (
                    (torch.sigmoid(preds[i][0, 0]) > 0.5)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                gt_2d = y[i].astype(np.uint8)

                pred_slices[i].append(pred_2d)
                gt_slices[i].append(gt_2d)

        # ── Stack along Z: [H, W, Z] ──
        pred_3d = [np.stack(slices, axis=-1) for slices in pred_slices]
        gt_3d = [np.stack(slices, axis=-1) for slices in gt_slices]

        return pred_3d, gt_3d

    def _compute_patient_metrics(
        self,
        pred_3d: List[np.ndarray],
        gt_3d: List[np.ndarray],
    ) -> Dict:
        """
        Compute all 3D metrics for one patient.

        Returns:
            {
              "lesion": {"dice_3d": float, "hd95": float},
              "lvo":    {"recall_3d": float, "object_f1": dict},
              "cow":    {"dice_3d": float, "hd95": float},
            }
        """
        results = {}

        # ── Lesion: Dice 3D + HD95 ──
        results["lesion"] = {
            "dice_3d": dice_3d(pred_3d[0], gt_3d[0]),
            "hd95": hausdorff_95(pred_3d[0], gt_3d[0], spacing=self.spacing),
        }

        # ── LVO: Recall 3D + Object F1 (centroid r=3) ──
        results["lvo"] = {
            "recall_3d": recall_3d(pred_3d[1], gt_3d[1]),
            "object_f1": object_f1_centroid(
                pred_3d[1], gt_3d[1], radius=self.centroid_radius
            ),
        }

        # ── CoW: Dice 3D + HD95 ──
        results["cow"] = {
            "dice_3d": dice_3d(pred_3d[2], gt_3d[2]),
            "hd95": hausdorff_95(pred_3d[2], gt_3d[2], spacing=self.spacing),
        }

        return results

    def _aggregate(self, per_patient: Dict) -> Dict[str, float]:
        """
        Aggregate per-patient metrics → mean ± std.

        Skips patients that failed evaluation (None values).
        """
        # Collect valid results
        valid = {pid: m for pid, m in per_patient.items() if m is not None}

        if not valid:
            logger.warning("No valid patient results to aggregate.")
            return {}

        summary = {}

        # ── Lesion ──
        lesion_dice = [v["lesion"]["dice_3d"] for v in valid.values()]
        lesion_hd95 = [
            v["lesion"]["hd95"]
            for v in valid.values()
            if v["lesion"]["hd95"] != float("inf")
        ]

        summary["lesion_dice_3d_mean"] = float(np.mean(lesion_dice))
        summary["lesion_dice_3d_std"] = float(np.std(lesion_dice))
        summary["lesion_hd95_mean"] = (
            float(np.mean(lesion_hd95)) if lesion_hd95 else float("inf")
        )
        summary["lesion_hd95_std"] = (
            float(np.std(lesion_hd95)) if lesion_hd95 else 0.0
        )

        # ── LVO ──
        lvo_recall = [v["lvo"]["recall_3d"] for v in valid.values()]
        lvo_f1 = [v["lvo"]["object_f1"]["f1"] for v in valid.values()]
        lvo_tp = sum(v["lvo"]["object_f1"]["tp"] for v in valid.values())
        lvo_fp = sum(v["lvo"]["object_f1"]["fp"] for v in valid.values())
        lvo_fn = sum(v["lvo"]["object_f1"]["fn"] for v in valid.values())

        summary["lvo_recall_3d_mean"] = float(np.mean(lvo_recall))
        summary["lvo_recall_3d_std"] = float(np.std(lvo_recall))
        summary["lvo_object_f1_mean"] = float(np.mean(lvo_f1))
        summary["lvo_object_f1_std"] = float(np.std(lvo_f1))
        summary["lvo_total_tp"] = lvo_tp
        summary["lvo_total_fp"] = lvo_fp
        summary["lvo_total_fn"] = lvo_fn

        # ── CoW ──
        cow_dice = [v["cow"]["dice_3d"] for v in valid.values()]
        cow_hd95 = [
            v["cow"]["hd95"]
            for v in valid.values()
            if v["cow"]["hd95"] != float("inf")
        ]

        summary["cow_dice_3d_mean"] = float(np.mean(cow_dice))
        summary["cow_dice_3d_std"] = float(np.std(cow_dice))
        summary["cow_hd95_mean"] = (
            float(np.mean(cow_hd95)) if cow_hd95 else float("inf")
        )
        summary["cow_hd95_std"] = (
            float(np.std(cow_hd95)) if cow_hd95 else 0.0
        )

        # ── Evaluated patients ──
        summary["n_patients_evaluated"] = len(valid)
        summary["n_patients_failed"] = len(per_patient) - len(valid)

        return summary

    def print_report(self, results: Dict):
        """Print formatted clinical evaluation report."""
        summary = results.get("summary", {})
        n_eval = summary.get("n_patients_evaluated", 0)
        n_fail = summary.get("n_patients_failed", 0)

        sep = "═" * 65
        print(f"\n{sep}")
        print("  ISLES'24 — 3D Volume Evaluation Report")
        print(f"{sep}")
        print(f"  Patients: {n_eval} evaluated, {n_fail} failed")
        print(f"  Spacing:  {self.spacing} mm")
        print(f"{'─' * 65}")

        # Lesion
        print(f"\n  🧠 LESION (Stroke Infarct)")
        print(
            f"     Dice 3D:   "
            f"{summary.get('lesion_dice_3d_mean', 0):.4f} "
            f"± {summary.get('lesion_dice_3d_std', 0):.4f}"
        )
        print(
            f"     HD95 (mm): "
            f"{summary.get('lesion_hd95_mean', 0):.2f} "
            f"± {summary.get('lesion_hd95_std', 0):.2f}"
        )

        # LVO
        print(f"\n  🩸 LVO (Large Vessel Occlusion)")
        print(
            f"     Recall 3D:   "
            f"{summary.get('lvo_recall_3d_mean', 0):.4f} "
            f"± {summary.get('lvo_recall_3d_std', 0):.4f}"
        )
        print(
            f"     Object F1:   "
            f"{summary.get('lvo_object_f1_mean', 0):.4f} "
            f"± {summary.get('lvo_object_f1_std', 0):.4f}"
        )
        print(
            f"     Detection:   "
            f"TP={summary.get('lvo_total_tp', 0)}, "
            f"FP={summary.get('lvo_total_fp', 0)}, "
            f"FN={summary.get('lvo_total_fn', 0)}"
        )

        # CoW
        print(f"\n  🔵 CoW (Circle of Willis)")
        print(
            f"     Dice 3D:   "
            f"{summary.get('cow_dice_3d_mean', 0):.4f} "
            f"± {summary.get('cow_dice_3d_std', 0):.4f}"
        )
        print(
            f"     HD95 (mm): "
            f"{summary.get('cow_hd95_mean', 0):.2f} "
            f"± {summary.get('cow_hd95_std', 0):.2f}"
        )

        print(f"\n{sep}\n")
