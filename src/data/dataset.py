"""
dataset.py — ISLES24Dataset: PyTorch Dataset with JSON-cached pos/neg
              slice classification and WeightedRandomSampler support.

Input:  [18, 544, 544] float16 on disk → float32 in memory
Label:  [3, 544, 544]  uint8 on disk   → float32 in memory
"""
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ISLES24Dataset(Dataset):
    """
    PyTorch Dataset for ISLES'24 2.5D NPY slices.

    Features:
      - Loads data on-the-fly from NVMe (136GB >> 30GB RAM)
      - JSON cache for positive/negative slice classification
      - Sample weights for WeightedRandomSampler
      - MONAI dictionary-based transforms
    """

    def __init__(
        self,
        patient_ids: List[str],
        patient_dirs: Dict[str, Path],
        transform: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        pos_weight: float = 3.0,
    ):
        """
        Args:
            patient_ids:  List of patient IDs assigned to this split
            patient_dirs: Dict mapping patient_id → absolute directory Path
            transform:    MONAI Compose pipeline (None for validation)
            cache_path:   Path to JSON cache for pos/neg slice info
            pos_weight:   Weight for positive slices in WeightedRandomSampler
        """
        super().__init__()
        self.transform = transform
        self.pos_weight = pos_weight

        # ── Build flat sample list: [(x_path, y_path), ...] ──
        self.samples: List[tuple] = []
        for pid in sorted(patient_ids):
            pdir = patient_dirs[pid]
            input_dir = pdir / "inputs"
            label_dir = pdir / "labels"

            for x_file in sorted(input_dir.glob("x_z*.npy")):
                # x_z000.npy → y_z000.npy
                z_suffix = x_file.stem[1:]           # "_z000"
                y_file = label_dir / f"y{z_suffix}.npy"

                if y_file.exists():
                    self.samples.append((str(x_file), str(y_file)))
                else:
                    logger.warning(f"Missing label for {x_file.name} in {pid}")

        logger.info(
            f"ISLES24Dataset: {len(self.samples)} slices "
            f"from {len(patient_ids)} patients"
        )

        # ── Build positive/negative slice cache ──
        self._is_positive: List[bool] = self._build_slice_cache(cache_path)

        n_pos = sum(self._is_positive)
        n_total = len(self._is_positive)
        logger.info(
            f"Slice balance: {n_pos} positive / {n_total - n_pos} negative "
            f"({100 * n_pos / max(n_total, 1):.1f}% positive)"
        )

    # ─────────────────────────────────────────────────────────────
    #  Cache logic
    # ─────────────────────────────────────────────────────────────

    def _build_slice_cache(self, cache_path: Optional[str]) -> List[Dict[str, bool]]:
        """
        Determine which slices contain which labels.
        Cache results to JSON for instant reload.
        """
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading slice cache from {cache_path}")
            with open(cache_path, "r") as f:
                return json.load(f)

        logger.info("Scanning labels for multi-task classification...")
        categories = []

        for i, (_, y_path) in enumerate(self.samples):
            y = np.load(y_path)
            cat = {
                "lesion": bool(y[0].any()),
                "lvo": bool(y[1].any()),
                "cow": bool(y[2].any())
            }
            cat["any_positive"] = cat["lesion"] or cat["lvo"] or cat["cow"]
            categories.append(cat)

            if (i + 1) % 2000 == 0:
                logger.info(f"  Scanned {i + 1}/{len(self.samples)} slices...")

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(categories, f)
            logger.info(f"Saved slice cache → {cache_path}")

        return categories

    # ─────────────────────────────────────────────────────────────
    #  PyTorch Dataset interface
    # ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x_path, y_path = self.samples[idx]

        # Load from disk and cast types
        x = np.load(x_path).astype(np.float32)     # float16 → float32 [18, 544, 544]
        y = np.load(y_path).astype(np.float32)     # uint8   → float32 [3, 544, 544]

        # ── 1. Clip Perfusion Outliers (Critical for Stability) ──
        # Perfusion channels are 6-17. Clip them to [-5.0, 5.0]
        x[6:18] = np.clip(x[6:18], -5.0, 5.0)

        # ── 2. Generate Brain Mask from NCCT (Channel 1 is central NCCT slice) ──
        # NCCT background is -1.0. We use > -0.95 to extract the brain.
        brain_mask = (x[1] > -0.95).astype(np.float32)  # [544, 544]
        brain_mask = np.expand_dims(brain_mask, axis=0) # [1, 544, 544]

        # Apply MONAI transforms (dict-based)
        if self.transform is not None:
            data = self.transform({"image": x, "label": y, "brain_mask": brain_mask})
            x, y, brain_mask = data["image"], data["label"], data["brain_mask"]

        # Ensure torch.Tensor output
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.ascontiguousarray(x))
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(np.ascontiguousarray(y))
        if not isinstance(brain_mask, torch.Tensor):
            brain_mask = torch.from_numpy(np.ascontiguousarray(brain_mask))

        return x, y, brain_mask

    # ─────────────────────────────────────────────────────────────
    #  Sampling support
    # ─────────────────────────────────────────────────────────────

    def get_sample_weights(self) -> np.ndarray:
        """
        Return weight array for torch.utils.data.WeightedRandomSampler.
        Implementing stratified sampling strategy from System Design Plan:
        - LVO -> 3.0 (Oversample 3x)
        - Lesion / CoW (Positive but no LVO) -> 1.0 (Keep 100%)
        - All negative -> 0.3 (Downsample to 30%)
        """
        weights = np.ones(len(self.samples), dtype=np.float64)
        for i, cat in enumerate(self._is_positive):
            if cat["lvo"]:
                weights[i] = 3.0
            elif cat["any_positive"]:
                weights[i] = 1.0
            else:
                weights[i] = 0.3
        return weights
