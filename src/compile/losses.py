"""
losses.py — Task-specific Loss Functions cho Multi-Task Stroke Segmentation

Loss assignment:
    Lesion: TverskyLoss(α=0.4, β=0.6)
        → Phạt FN nặng hơn FP — không bỏ sót vùng nhồi máu

    LVO:    FocalTverskyLoss(α=0.2, β=0.8, γ=3.0)
        → Focus cực mạnh vào hard samples — LVO rất nhỏ và rất hiếm
        → β=0.8: Bỏ sót LVO = bệnh nhân mất cơ hội can thiệp

    CoW:    TverskyLoss(α=0.5, β=0.5)
        → Cân bằng — giải phẫu mạch máu lớn, ít biến động

Tất cả input là RAW LOGITS (chưa sigmoid).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Tversky Loss ─────────────────────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    Tversky Loss — tổng quát hóa của Dice Loss.

    L_Tversky = 1 - TP / (TP + α·FP + β·FN)

    Khi α = β = 0.5 → Dice Loss.
    Khi β > α → phạt FN nặng hơn (bỏ sót vùng bệnh nguy hiểm hơn đoán nhầm).

    Args:
        alpha:   Trọng số phạt False Positive
        beta:    Trọng số phạt False Negative
        smooth:  Epsilon để tránh division by zero
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 1, H, W) — raw logits
            targets: (B, 1, H, W) — binary mask {0.0, 1.0}

        Returns:
            Scalar loss
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dims để tính tổng
        probs   = probs.view(probs.size(0), -1)    # (B, H*W)
        targets = targets.view(targets.size(0), -1)  # (B, H*W)

        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1 - targets)).sum(dim=1)
        FN = ((1 - probs) * targets).sum(dim=1)

        # Sử dụng smooth lớn hơn (1e-5) cho float16 stability
        numerator = TP + self.smooth
        denominator = TP + self.alpha * FP + self.beta * FN + self.smooth
        
        # Clamp denominator để tránh chia cho 0 tuyệt đối
        tversky_index = numerator / denominator.clamp(min=self.smooth)
        
        return 1.0 - tversky_index.mean()


# ─── Focal Tversky Loss ───────────────────────────────────────────────────────

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss — thêm hệ số γ để focus vào hard samples.

    L_FocalTversky = (1 - TverskyIndex)^γ

    Khi γ > 1 → loss tại các vùng khó (small/rare lesions) được khuếch đại.
    γ = 3.0 cho LVO: Ép mô hình tập trung vào những điểm tắc mạch nhỏ khó detect.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float  = 0.5,
        gamma: float = 2.0,
        smooth: float = 1e-5, # Tăng nhẹ để ổn định AMP
    ):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits)
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1 - targets)).sum(dim=1)
        FN = ((1 - probs) * targets).sum(dim=1)

        numerator = TP + self.smooth
        denominator = TP + self.alpha * FP + self.beta * FN + self.smooth
        
        tversky_index = numerator / denominator.clamp(min=self.smooth)
        
        # focal_tversky = (1 - TI)^gamma
        # Dùng clamp an toàn hơn cho float16
        error = (1.0 - tversky_index).clamp(min=1e-6, max=1.0)
        focal_tversky = torch.pow(error, self.gamma)
        
        return focal_tversky.mean()


# ─── Multi-Task Loss ──────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Tổng hợp loss từ 3 tasks với trọng số riêng.

    L_total = w_lesion * L_lesion + w_lvo * L_lvo + w_cow * L_cow

    w_lvo = 10.0 vì phát hiện LVO là nhiệm vụ cấp cứu — sai lầm = tử vong.
    """

    def __init__(self, config: dict):
        super().__init__()
        loss_cfg = config["loss"]

        # Lesion: Tversky
        lesion_cfg = loss_cfg["lesion"]
        self.lesion_loss = TverskyLoss(
            alpha=lesion_cfg["alpha"],
            beta=lesion_cfg["beta"],
        )
        self.w_lesion = lesion_cfg["weight"]

        # LVO: Focal Tversky
        lvo_cfg = loss_cfg["lvo"]
        self.lvo_loss = FocalTverskyLoss(
            alpha=lvo_cfg["alpha"],
            beta=lvo_cfg["beta"],
            gamma=lvo_cfg["gamma"],
        )
        self.w_lvo = lvo_cfg["weight"]

        # CoW: Tversky (balanced)
        cow_cfg = loss_cfg["cow"]
        self.cow_loss = TverskyLoss(
            alpha=cow_cfg["alpha"],
            beta=cow_cfg["beta"],
        )
        self.w_cow = cow_cfg["weight"]

    def forward(self, preds: dict, targets: torch.Tensor) -> dict:
        """
        Args:
            preds:   dict {'lesion': (B,1,H,W), 'lvo': (B,1,H,W), 'cow': (B,1,H,W)} — logits
            targets: (B, 3, H, W) — binary masks [lesion, lvo, cow]

        Returns:
            dict với keys: 'total', 'lesion', 'lvo', 'cow'
        """
        l_lesion = self.lesion_loss(preds["lesion"], targets[:, 0:1])
        l_lvo    = self.lvo_loss(preds["lvo"],       targets[:, 1:2])
        l_cow    = self.cow_loss(preds["cow"],        targets[:, 2:3])

        total = self.w_lesion * l_lesion + self.w_lvo * l_lvo + self.w_cow * l_cow

        return {
            "total":  total,
            "lesion": l_lesion,
            "lvo":    l_lvo,
            "cow":    l_cow,
        }
