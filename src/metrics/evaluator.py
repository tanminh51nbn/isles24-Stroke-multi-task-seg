import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .functional import dice_3d, recall_3d, object_f1_centroid

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Post-training 3D evaluation pipeline.
    Reconstructs full 3D volumes and calculates medical-grade metrics.
    """
    TASK_NAMES = ("lesion", "lvo", "cow")

    def __init__(
        self,
        model: nn.Module,
        patient_dirs: List[Path],
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.patient_dirs = patient_dirs
        self.device = device

    @torch.no_grad()
    def evaluate_all(self):
        """Runs evaluation over all patients."""
        metrics = {"lesion_dice": [], "lvo_f1": [], "cow_dice": []}
        
        logger.info(f"Starting 3D Evaluation for {len(self.patient_dirs)} patients.")
        
        for pdir in tqdm(self.patient_dirs, desc="Evaluating Patients"):
            res = self.evaluate_patient(pdir)
            if res:
                metrics["lesion_dice"].append(res["lesion_dice"])
                metrics["lvo_f1"].append(res["lvo_f1"])
                metrics["cow_dice"].append(res["cow_dice"])
                
        final = {
            "lesion_dice": np.mean(metrics["lesion_dice"]) if metrics["lesion_dice"] else 0,
            "lvo_f1": np.mean(metrics["lvo_f1"]) if metrics["lvo_f1"] else 0,
            "cow_dice": np.mean(metrics["cow_dice"]) if metrics["cow_dice"] else 0,
        }
        
        logger.info("--- Final 3D Metrics ---")
        logger.info(f"Lesion Dice: {final['lesion_dice']:.4f}")
        logger.info(f"LVO F1:      {final['lvo_f1']:.4f}")
        logger.info(f"CoW Dice:    {final['cow_dice']:.4f}")
        
        return final

    def evaluate_patient(self, pdir: Path) -> dict:
        img_dir = pdir / "images"
        lbl_dir = pdir / "labels"
        
        if not img_dir.exists() or not lbl_dir.exists():
            return None
            
        slice_files = sorted(list(img_dir.glob("slice_*.npy")))
        if not slice_files:
            return None
            
        preds_3d = {t: [] for t in self.TASK_NAMES}
        targets_3d = {t: [] for t in self.TASK_NAMES}
        
        for sf in slice_files:
            lf = lbl_dir / sf.name
            
            # Load
            img = np.load(sf).astype(np.float32)
            lbl = np.load(lf).astype(np.float32)
            
            # Clip
            img[6:18] = np.clip(img[6:18], -5.0, 5.0)
            
            # Brain mask
            brain_mask = (img[1] > -0.95).astype(np.float32)
            
            # To tensor
            x = torch.from_numpy(img).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(brain_mask).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(x)
                
            for i, t in enumerate(self.TASK_NAMES):
                # Apply mask
                logits_i = logits[i] * mask_t + (-1e9) * (1 - mask_t)
                pred_bin = (torch.sigmoid(logits_i) > 0.5).float().cpu().numpy()[0, 0]
                
                preds_3d[t].append(pred_bin)
                targets_3d[t].append(lbl[i])
                
        # Stack to 3D: [Z, H, W]
        p_vol = {t: np.stack(preds_3d[t]) for t in self.TASK_NAMES}
        t_vol = {t: np.stack(targets_3d[t]) for t in self.TASK_NAMES}
        
        return {
            "lesion_dice": dice_3d(p_vol["lesion"], t_vol["lesion"]),
            "lvo_f1": object_f1_centroid(p_vol["lvo"], t_vol["lvo"], radius=3),
            "cow_dice": dice_3d(p_vol["cow"], t_vol["cow"])
        }
