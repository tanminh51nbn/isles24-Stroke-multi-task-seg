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
from monai.losses import DiceFocalLoss

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for simultaneous Lesion/LVO/CoW segmentation.

    Each task gets its own DiceFocalLoss computation, then losses
    are aggregated with configurable task weights.
    """

    TASK_NAMES = ("lesion", "lvo", "cow")

    def __init__(self, loss_cfg: dict, task_weights: dict):
        """
        Args:
            loss_cfg:     loss section from configs/train.yaml
            task_weights: {"lesion": 1.0, "lvo": 2.0, "cow": 1.0}
        """
        super().__init__()

        # ── MONAI DiceFocalLoss ──
        self.criterion = DiceFocalLoss(
            sigmoid=loss_cfg.get("sigmoid", True),
            include_background=True,               # ⚠️ Must be True for [B,1,H,W]
            lambda_dice=loss_cfg.get("lambda_dice", 1.0),
            lambda_focal=loss_cfg.get("lambda_focal", 1.0),
            gamma=loss_cfg.get("gamma", 2.0),
        )

        # ── Task weights ──
        self.task_weights = {
            name: task_weights.get(name, 1.0)
            for name in self.TASK_NAMES
        }

        logger.info(
            f"MultiTaskLoss: DiceFocal(gamma={loss_cfg.get('gamma', 2.0)}) | "
            f"weights={self.task_weights}"
        )

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

            task_loss = self.criterion(pred, target)
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
