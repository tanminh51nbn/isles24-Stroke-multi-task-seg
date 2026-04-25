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
        accelerator = None,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.patient_dirs = patient_dirs
        self.device = device
        self.accelerator = accelerator

    @torch.no_grad()
    def evaluate_all(self):
        """Runs evaluation over all patients using distributed strategy if available."""
        # --- Distributed Split ---
        all_patients = self.patient_dirs
        rank = 0
        if self.accelerator is not None:
            rank = self.accelerator.process_index
            world_size = self.accelerator.num_processes
            local_patients = all_patients[rank::world_size]
        else:
            local_patients = all_patients

        local_metrics = {"lesion_dice": [], "lvo_f1": [], "cow_dice": []}
        
        pbar = tqdm(local_patients, desc=f"GPU {rank} Evaluating", disable=not (rank == 0 or self.accelerator is None))
        
        for pdir in pbar:
            res = self.evaluate_patient(pdir)
            if res:
                local_metrics["lesion_dice"].append(res["lesion_dice"])
                local_metrics["lvo_f1"].append(res["lvo_f1"])
                local_metrics["cow_dice"].append(res["cow_dice"])
        
        local_sums = torch.tensor([
            np.sum(local_metrics["lesion_dice"]) if local_metrics["lesion_dice"] else 0.0,
            np.sum(local_metrics["lvo_f1"]) if local_metrics["lvo_f1"] else 0.0,
            np.sum(local_metrics["cow_dice"]) if local_metrics["cow_dice"] else 0.0,
            float(len(local_metrics["lesion_dice"]))
        ], device=self.device)

        if self.accelerator is not None:
            # Gather [Sum_Les, Sum_LVO, Sum_CoW, Count] from all GPUs
            all_res = self.accelerator.gather(local_sums.unsqueeze(0)) # [world_size, 4]
            global_sums = all_res.sum(dim=0).cpu().numpy()
            total_count = global_sums[3]
            
            final = {
                "lesion_dice": global_sums[0] / max(total_count, 1),
                "lvo_f1":      global_sums[1] / max(total_count, 1),
                "cow_dice":    global_sums[2] / max(total_count, 1),
            }
        else:
            total_count = local_sums[3].item()
            final = {
                "lesion_dice": local_sums[0].item() / max(total_count, 1),
                "lvo_f1":      local_sums[1].item() / max(total_count, 1),
                "cow_dice":    local_sums[2].item() / max(total_count, 1),
            }
        
        if rank == 0:
            logger.info("--- Final 3D Metrics (Aggregated) ---")
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
        
        # Mini-batch Inference to avoid OOM
        inf_bs = 16
        preds_3d = {t: [] for t in self.TASK_NAMES}
        
        with torch.no_grad():
            for i in range(0, x_batch.shape[0], inf_bs):
                x_sub = x_batch[i : i + inf_bs]
                m_sub = mask_t[i : i + inf_bs]
                
                with torch.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", dtype=torch.float16):
                    logits = self.model(x_sub) # tuple of 3
                
                for t_idx, t in enumerate(self.TASK_NAMES):
                    l_sub = logits[t_idx] * m_sub + (-1e9) * (1 - m_sub)
                    pred_bin = (torch.sigmoid(l_sub) > 0.5).byte().cpu().numpy()[:, 0]
                    preds_3d[t].append(pred_bin)
                    
        # Concatenate results back to full volume
        for t in self.TASK_NAMES:
            preds_3d[t] = np.concatenate(preds_3d[t], axis=0)
        
        del x_batch, mask_t, logits
        gc.collect() 
        
        # Final Volumes: [Z, H, W] in uint8
        t_vol = np.stack(all_lbls) # [N, 3, H, W]
        
        results = {
            "lesion_dice": dice_3d(preds_3d["lesion"], t_vol[:, 0]),
            "lvo_f1": object_f1_centroid(preds_3d["lvo"], t_vol[:, 1], radius=3),
            "cow_dice": dice_3d(preds_3d["cow"], t_vol[:, 2])
        }
        
        del preds_3d, t_vol, all_lbls
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        gc.collect()
        return results
