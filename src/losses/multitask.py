import torch
import torch.nn as nn
from .tversky import build_lesion_loss
from .focal_tversky import build_lvo_loss
from .dice_focal import build_cow_loss

class MultiTaskLoss(nn.Module):
    """
    Combines Lesion, LVO, and CoW losses with configurable weights.
    """
    TASK_NAMES = ("lesion", "lvo", "cow")

    def __init__(self, cfg: dict):
        super().__init__()
        
        # Load weights from config
        weights_cfg = cfg.get("task_weights", {})
        self.weights = {
            "lesion": weights_cfg.get("lesion", 1.0),
            "lvo": weights_cfg.get("lvo", 3.0),
            "cow": weights_cfg.get("cow", 0.8),
        }
        
        # Criterions - Read detailed params from loss section
        loss_cfg = cfg.get("loss", {})
        les_p = loss_cfg.get("lesion", {})
        lvo_p = loss_cfg.get("lvo", {})
        cow_p = loss_cfg.get("cow", {})

        self.criterions = nn.ModuleDict({
            "lesion": build_lesion_loss(
                alpha=les_p.get("alpha", 0.4), 
                beta=les_p.get("beta", 0.6)
            ),
            "lvo": build_lvo_loss(
                alpha=lvo_p.get("alpha", 0.2), 
                beta=lvo_p.get("beta", 0.8), 
                gamma=lvo_p.get("gamma", 2.0)
            ),
            "cow": build_cow_loss(
                alpha=cow_p.get("alpha", 0.5), 
                beta=cow_p.get("beta", 0.5)
            )
        })

    def forward(self, preds, y, brain_mask=None):
        """
        preds: tuple of 3 logits [B, 1, H, W]
        y: target [B, 3, H, W]
        brain_mask: binary mask [B, 1, H, W] (Redundant, applied in trainer)
        """
        loss_dict = {}
        total_loss = None
        
        for i, task in enumerate(self.TASK_NAMES):
            task_logits = preds[i]
            task_target = y[:, i:i+1]
            
            # Compute loss
            task_loss = self.criterions[task](task_logits, task_target)
            weighted = task_loss * self.weights[task]
            
            # Logging
            loss_dict[task] = task_loss.detach().item()
            
            # Accumulate total loss
            if total_loss is None:
                total_loss = weighted
            else:
                total_loss = total_loss + weighted
            
        return total_loss, loss_dict

def build_loss(cfg: dict) -> nn.Module:
    return MultiTaskLoss(cfg)
