"""
multi_task_loss.py — Multi-Task DiceFocal Loss for ISLES'24.

Combines MONAI's DiceFocalLoss with per-task weighting:
    Loss_total = w_lesion · L_lesion + w_lvo · L_lvo + w_cow · L_cow

Design notes:
  - sigmoid=True: loss function applies sigmoid to raw logits internally
  - include_background=True: MUST be True for 1-channel binary output
  - gamma=2.0: Focal Loss focuses on hard pixels (lesion boundaries, tiny LVO)
    ► Fine-tuning: increase to 3.0 if LVO recall is low (see references below)

References for gamma tuning:
  - Lin et al. (2017) "Focal Loss for Dense Object Detection": gamma=2.0 optimal
  - Medical imaging studies typically use gamma ∈ [2.0, 3.0]
  - Strategy: start gamma=2.0, increase to 3.0 in fine-tuning if recall on
    small objects (LVO) remains low. Avoid gamma > 5.0 (unstable gradients).
"""
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss, FocalLoss, TverskyLoss

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for simultaneous Lesion/LVO/CoW segmentation.

    Each task gets its own specialized loss (e.g., FocalTversky for imbalanced LVO).
    """

    TASK_NAMES = ("lesion", "lvo", "cow")

    def __init__(self, loss_cfg: dict, task_weights: dict):
        """
        Args:
            loss_cfg:     loss section from configs/train.yaml
            task_weights: {"lesion": 2.0, "lvo": 5.0, "cow": 1.0}
        """
        super().__init__()
        
        self.criterions = nn.ModuleDict()
        
        for task_name in self.TASK_NAMES:
            cfg = loss_cfg.get(task_name, {})
            loss_type = cfg.get("type", "dice_focal")
            
            if loss_type == "dice_focal":
                self.criterions[task_name] = DiceFocalLoss(
                    sigmoid=True,
                    include_background=True,
                    lambda_dice=cfg.get("lambda_dice", 1.0),
                    lambda_focal=cfg.get("lambda_focal", 1.0),
                    gamma=cfg.get("gamma", 2.0),
                )
            elif loss_type == "focal_tversky":
                # MONAI's TverskyLoss has an internal focal parameter if needed,
                # but we can use TverskyLoss + Focal behavior.
                # Actually, MONAI FocalLoss and TverskyLoss are separate, but 
                # TverskyLoss has no built-in focal power. We can use TverskyLoss 
                # and apply focal manually, or just use TverskyLoss as is.
                # Wait, MONAI has FocalLoss, but no FocalTverskyLoss directly.
                # Let's implement Focal Tversky natively here.
                pass # Handled below
                
        # To handle Focal Tversky cleanly without relying on MONAI internals, 
        # let's just initialize the standard TverskyLoss and we will apply focal in forward.
        for task_name in self.TASK_NAMES:
            cfg = loss_cfg.get(task_name, {})
            loss_type = cfg.get("type", "dice_focal")
            
            if loss_type == "focal_tversky":
                self.criterions[task_name] = TverskyLoss(
                    sigmoid=True,
                    include_background=True,
                    alpha=cfg.get("alpha", 0.5),
                    beta=cfg.get("beta", 0.5),
                )
                
        # Store focal gammas separately
        self.gammas = {
            name: loss_cfg.get(name, {}).get("gamma", 1.0)
            for name in self.TASK_NAMES
        }
        self.loss_types = {
            name: loss_cfg.get(name, {}).get("type", "dice_focal")
            for name in self.TASK_NAMES
        }

        # ── Task weights ──
        self.task_weights = {
            name: task_weights.get(name, 1.0)
            for name in self.TASK_NAMES
        }

        logger.info(f"MultiTaskLoss Task Weights: {self.task_weights}")
        logger.info(f"Loss Types: {self.loss_types}")

    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted multi-task loss.

        Args:
            predictions: list of 3 tensors, each [B, 1, H, W] (raw logits)
                         Order: [lesion, lvo, cow]
            targets:     [B, 3, H, W] float32 binary masks

        Returns:
            total_loss: scalar tensor (backprop-ready, has grad)
            loss_dict:  {"lesion": float, "lvo": float, "cow": float, "total": float}
                        (detached values for logging)
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=predictions[0].device)

        for i, task_name in enumerate(self.TASK_NAMES):
            pred = predictions[i]                   # [B, 1, H, W] logits
            target = targets[:, i : i + 1, :, :]    # [B, 1, H, W] binary

            criterion = self.criterions[task_name]
            task_loss = criterion(pred, target)
            
            if self.loss_types[task_name] == "focal_tversky":
                gamma = self.gammas[task_name]
                # Focal Tversky formulation: (TverskyLoss)^gamma
                # Note: TverskyLoss computes 1 - TverskyIndex. 
                # So task_loss is already (1 - TI). We just raise it to gamma.
                task_loss = torch.pow(task_loss + 1e-6, 1.0 / gamma) # wait, usually it's (1 - TI)^(1/gamma) in MONAI implementation of Focal Tversky, or just task_loss ** gamma. Let's use (task_loss)**gamma for focusing. 
                # Actually standard Focal Tversky is: TL = (1 - TverskyIndex)^(1/gamma) according to Abraham et al. 
                # Let's use 1/gamma.
                task_loss = torch.pow(task_loss + 1e-6, 1.0 / gamma)

            weighted = self.task_weights[task_name] * task_loss

            loss_dict[task_name] = task_loss.item()
            total_loss = total_loss + weighted

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


# ═══════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════

def build_loss(cfg: dict) -> MultiTaskLoss:
    """
    Build MultiTaskLoss from config.

    Args:
        cfg: full training config dict (uses cfg["loss"] and cfg["task_weights"])

    Returns:
        Initialized MultiTaskLoss module
    """
    return MultiTaskLoss(
        loss_cfg=cfg["loss"],
        task_weights=cfg["task_weights"],
    )
