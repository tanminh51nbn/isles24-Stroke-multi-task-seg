import torch
import torch.nn as nn
from monai.losses import TverskyLoss

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for LVO Segmentation.
    Handle extreme imbalance + penalize FN.
    """
    def __init__(self, alpha: float = 0.2, beta: float = 0.8, gamma: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.tversky = TverskyLoss(
            include_background=True, 
            sigmoid=True, 
            alpha=alpha, 
            beta=beta
        )
        self.gamma = gamma

    def forward(self, logits, target):
        tversky_loss = self.tversky(logits, target)
        return torch.pow(tversky_loss + 1e-6, self.gamma)

def build_lvo_loss(alpha: float = 0.2, beta: float = 0.8, gamma: float = 2.0):
    return FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
