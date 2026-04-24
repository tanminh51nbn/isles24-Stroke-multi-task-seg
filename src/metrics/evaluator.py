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
        import gc
        img_dir = pdir / "inputs"
        lbl_dir = pdir / "labels"
        
        if not img_dir.exists() or not lbl_dir.exists():
            return None
            
        slice_files = sorted(list(img_dir.glob("x_z*.npy")))
        if not slice_files:
            return None
            
        # --- Batch Inference Optimization ---
        all_imgs = []
        all_lbls = []
        
        for sf in slice_files:
            lf = lbl_dir / sf.name.replace("x_z", "y_z")
            all_imgs.append(np.load(sf).astype(np.float32))
            all_lbls.append(np.load(lf).astype(np.uint8)) # Labels are binary uint8
            
        # Stack to tensors: [N, C, H, W]
        x_batch = torch.from_numpy(np.stack(all_imgs)).to(self.device)
        del all_imgs # Free CPU RAM immediately
        
        # GPU Preprocessing
        x_batch[:, 6:18] = torch.clamp(x_batch[:, 6:18], -5.0, 5.0)
        mask_t = (x_batch[:, 1:2] > -0.95).float()
        
        preds_3d = {t: [] for t in self.TASK_NAMES}
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # Run whole patient in one GPU call (Safe for T4)
                logits = self.model(x_batch)
                logits = list(logits)
                
            for i, t in enumerate(self.TASK_NAMES):
                # Apply mask & threshold
                logits_i = logits[i] * mask_t + (-1e9) * (1 - mask_t)
                pred_bin = (torch.sigmoid(logits_i) > 0.5).byte() # Use byte (uint8) for RAM
                preds_3d[t] = pred_bin.cpu().numpy()[:, 0] # [N, H, W]
                del logits_i
        
        del x_batch, mask_t, logits
        gc.collect() # Trigger garbage collection
        
        # Final Volumes: [Z, H, W] in uint8
        t_vol = np.stack(all_lbls) # [N, 3, H, W]
        
        results = {
            "lesion_dice": dice_3d(preds_3d["lesion"], t_vol[:, 0]),
            "lvo_f1": object_f1_centroid(preds_3d["lvo"], t_vol[:, 1], radius=3),
            "cow_dice": dice_3d(preds_3d["cow"], t_vol[:, 2])
        }
        
        del preds_3d, t_vol, all_lbls
        return results
