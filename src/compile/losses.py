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

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1 - targets)).sum(dim=1)
        FN = ((1 - probs) * targets).sum(dim=1)

        numerator   = TP + self.smooth
        denominator = TP + self.alpha * FP + self.beta * FN + self.smooth
        tversky_index = numerator / denominator.clamp(min=self.smooth)

        return (1.0 - tversky_index.mean()).clamp(min=0.0, max=1.0)


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

        numerator   = TP + self.smooth
        denominator = TP + self.alpha * FP + self.beta * FN + self.smooth
        tversky_index = numerator / denominator.clamp(min=self.smooth)

        # Clamp error trong [1e-6, 1.0] trước khi pow() để tránh float16 overflow
        # float16 max = 65504. pow(1.0, 3.0) = 1.0 (safe). pow(1.0001, 3.0) có thể inf.
        error = (1.0 - tversky_index).clamp(min=1e-6, max=1.0)
        focal_tversky = torch.pow(error, self.gamma)

        # Clamp output cuối để đảm bảo loss luôn hữu hạn kể cả khi AMP overflow
        return focal_tversky.mean().clamp(max=10.0)


# ─── Multi-Task Loss ──────────────────────────────────────────────────────────

# ─── Boundary Loss (Soft Boundary Dice) ───────────────────────────────────────

class BoundaryLoss(nn.Module):
    """
    Boundary Loss — Phạt các lỗi tại đường biên của vật thể.
    Tận dụng MaxPool để tìm boundary của nhãn và dự đoán một cách vi phân.
    
    L_boundary = 1 - 2*|P_b ∩ T_b| / (|P_b| + |T_b|)
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Tách biên của nhãn (Boundary = Target - Erode(Target))
        # Erode(T) ≈ 1 - MaxPool(1 - T)
        targets_boundary = targets - (1.0 - F.max_pool2d(1.0 - targets, kernel_size=self.kernel_size, stride=1, padding=self.padding))
        targets_boundary = torch.clamp(targets_boundary, min=0.0)
        
        # Tách biên của dự đoán (Soft boundary)
        probs_boundary = probs - (1.0 - F.max_pool2d(1.0 - probs, kernel_size=self.kernel_size, stride=1, padding=self.padding))
        probs_boundary = torch.clamp(probs_boundary, min=0.0)
        
        # Dice trên vùng biên
        intersection = (probs_boundary * targets_boundary).sum(dim=(1, 2, 3))
        union = probs_boundary.sum(dim=(1, 2, 3)) + targets_boundary.sum(dim=(1, 2, 3))
        
        boundary_dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        return 1.0 - boundary_dice.mean()


# ─── Multi-Task Loss ──────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Tổng hợp loss từ 3 tasks với trọng số riêng.
    Hỗ trợ kết hợp Tversky/FocalTversky với Boundary Loss.
    """

    def __init__(self, config: dict):
        super().__init__()
        loss_cfg = config["loss"]

        # 1. Lesion Task
        lesion_cfg = loss_cfg["lesion"]
        self.lesion_main_loss = TverskyLoss(alpha=lesion_cfg["alpha"], beta=lesion_cfg["beta"])
        self.w_lesion = lesion_cfg["weight"]
        self.w_lesion_boundary = lesion_cfg.get("boundary_weight", 0.0)

        # 2. LVO Task
        lvo_cfg = loss_cfg["lvo"]
        self.lvo_main_loss = FocalTverskyLoss(alpha=lvo_cfg["alpha"], beta=lvo_cfg["beta"], gamma=lvo_cfg["gamma"])
        self.w_lvo = lvo_cfg["weight"]
        self.w_lvo_boundary = lvo_cfg.get("boundary_weight", 0.0)

        # 3. CoW Task
        cow_cfg = loss_cfg["cow"]
        self.cow_main_loss = TverskyLoss(alpha=cow_cfg["alpha"], beta=cow_cfg["beta"])
        self.w_cow = cow_cfg["weight"]
        self.w_cow_boundary = cow_cfg.get("boundary_weight", 0.0)

        # Công cụ Boundary Loss dùng chung
        self.boundary_loss = BoundaryLoss(kernel_size=3)

    def forward(self, preds: dict, targets: torch.Tensor) -> dict:
        # 1. Tính toán Main Loss (256x256) như cũ
        l_lesion_main = self.lesion_main_loss(preds["lesion"], targets[:, 0:1])
        l_lesion_bd   = self.boundary_loss(preds["lesion"], targets[:, 0:1])
        l_lesion      = (1.0 - self.w_lesion_boundary) * l_lesion_main + self.w_lesion_boundary * l_lesion_bd

        l_lvo_main = self.lvo_main_loss(preds["lvo"], targets[:, 1:2])
        l_lvo_bd   = self.boundary_loss(preds["lvo"], targets[:, 1:2])
        l_lvo      = (1.0 - self.w_lvo_boundary) * l_lvo_main + self.w_lvo_boundary * l_lvo_bd

        l_cow_main = self.cow_main_loss(preds["cow"], targets[:, 2:3])
        l_cow_bd   = self.boundary_loss(preds["cow"], targets[:, 2:3])
        l_cow      = (1.0 - self.w_cow_boundary) * l_cow_main + self.w_cow_boundary * l_cow_bd

        main_loss = self.w_lesion * l_lesion + self.w_lvo * l_lvo + self.w_cow * l_cow

        # 2. Tính toán Auxiliary Losses (MDS)
        # aux_masks: [mask_32, mask_64, mask_128, mask_256]
        aux_loss = 0.0
        # Trọng số tăng dần, giảm áp lực ở tầng quá sâu để tránh nhiễu
        aux_weights = [0.05, 0.075, 0.125, 0.25] 
        
        if "aux_masks" in preds and preds["aux_masks"] is not None:
            for i, aux_pred in enumerate(preds["aux_masks"]):
                if aux_pred is None or i >= len(aux_weights): continue
                
                # Resize nhãn GT xuống kích thước của aux_pred
                h, w = aux_pred.shape[2], aux_pred.shape[3]
                aux_targets = F.interpolate(targets, size=(h, w), mode="nearest")
                
                # --- Selective Supervision Logic ---
                # Luôn tính Lesion và CoW ở mọi tầng
                l_lesion_aux = self.lesion_main_loss(aux_pred[:, 0:1], aux_targets[:, 0:1])
                l_cow_aux    = self.cow_main_loss(aux_pred[:, 2:3], aux_targets[:, 2:3])
                
                # Chỉ tính LVO ở tầng 128 (i=2) và 256 (i=3)
                # Lý do: Ở 32 và 64, điểm LVO bị biến mất do nén ảnh, gây nhiễu gradient.
                if i >= 2:
                    l_lvo_aux = self.lvo_main_loss(aux_pred[:, 1:2], aux_targets[:, 1:2])
                    l_aux = self.w_lesion * l_lesion_aux + self.w_lvo * l_lvo_aux + self.w_cow * l_cow_aux
                else:
                    # Ở tầng sâu, chỉ tập trung vào Lesion và CoW (re-scale trọng số)
                    w_sum = self.w_lesion + self.w_cow
                    l_aux = (self.w_lesion / w_sum) * l_lesion_aux + (self.w_cow / w_sum) * l_cow_aux
                
                aux_loss += aux_weights[i] * l_aux

        # 3. Tổng hợp
        total = main_loss + aux_loss
        total = torch.nan_to_num(total, nan=1.0, posinf=1.0, neginf=1.0)

        return {
            "total":  total,
            "main":   main_loss,
            "aux":    aux_loss,
            "lesion": l_lesion,
            "lvo":    l_lvo,
            "cow":    l_cow,
            "boundary": (l_lesion_bd + l_lvo_bd + l_cow_bd) / 3.0
        }
