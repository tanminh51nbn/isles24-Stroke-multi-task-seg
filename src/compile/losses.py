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
import math


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


# ─── Modified Focal Loss (CenterNet-style Heatmap Loss) ─────────────────────

class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss dành riêng cho Heatmap Regression (LVO).
    Dựa trên paper CenterNet (Objects as Points, 2019).

    Công thức:
        Tại tâm (y = 1.0):  L = -(1 - pred)^alpha * log(pred)
        Tại vùng nền (y < 1): L = -(1 - y)^beta * pred^alpha * log(1 - pred)

    - Giảm phạt vùng "suýt đúng" (gần tâm, gt ~ 0.8): (1-0.8)^4 = 0.0016
    - Phạt nặng khi bỏ sót tâm thật sự (gt=1, pred=0): (1-0)^2 * log(0+eps)
    - Loss luôn nằm trong [0, 1] sau khi clamp để AMP an toàn.

    Args:
        alpha: Hệ số Focal cho vùng tâm (mặc định 2).
        beta:  Hệ số giảm phạt cho vùng quầng sáng xung quanh (mặc định 4).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    def forward(self, logits: torch.Tensor, heatmap_gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:     Raw logits từ LVO head, shape (B, 1, H, W)
            heatmap_gt: Gaussian Heatmap GT, shape (B, 1, H, W), values [0, 1]
        """
        # [QUAN TRỌNG] Chuyển sang FP32 để tránh tràn số (overflow) và NaN khi cộng tổng (sum) trong AMP
        logits = logits.float()
        heatmap_gt = heatmap_gt.float()

        pred = torch.sigmoid(logits)
        pred = pred.clamp(min=self.eps, max=1.0 - self.eps)

        # Phân vùng tâm (gt = 1.0) và vùng không phải tâm (gt < 1.0)
        pos_mask = (heatmap_gt == 1.0).float()
        neg_mask = 1.0 - pos_mask

        # Loss tại tâm: phạt nặng khi pred thấp
        pos_loss = -pos_mask * torch.pow(1.0 - pred, self.alpha) * torch.log(pred)

        # Loss tại nền: giảm phạt tỉ lệ với (1 - gt)^beta — càng gần tâm, phạt càng nhẹ
        neg_loss = -neg_mask * torch.pow(1.0 - heatmap_gt, self.beta) * \
                   torch.pow(pred, self.alpha) * torch.log(1.0 - pred)

        # Tính tổng theo từng ảnh trong batch (dim H, W)
        pos_loss = pos_loss.sum(dim=(1, 2, 3))
        neg_loss = neg_loss.sum(dim=(1, 2, 3))
        
        # Số pixel tâm thực sự (tránh chia 0 khi không có LVO trong ảnh)
        num_pos = pos_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)

        # Trọng số trung bình của batch
        # KHÔNG ĐƯỢC CLAMP TỔNG LOSS (sẽ làm gradient = 0)
        loss = ((pos_loss + neg_loss) / num_pos).mean()
        
        return loss


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

class SDFBoundaryLoss(nn.Module):
    """
    Hausdorff-inspired Boundary Loss dựa trên Signed Distance Function (SDF).
    Paper: "Boundary loss for highly unbalanced segmentation" (Kervadec et al.)
    
    Công thức: L = mean( pred * sdf )
    - Pixel sai ở xa ranh giới (sdf lớn) sẽ bị phạt rất nặng.
    - Pixel đúng (pred=1, sdf âm) sẽ làm giảm loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Dự đoán từ mô hình (B, 1, H, W)
            sdf:    Bản đồ khoảng cách từ ground truth (B, 1, H, W)
        """
        probs = torch.sigmoid(logits)
        
        # Loss = mean( probs * sdf )
        # Lưu ý: sdf bên ngoài dương, bên trong âm.
        # Dự đoán đúng vùng trong (probs=1 * sdf=-5) -> Giảm loss.
        # Dự đoán sai vùng ngoài (probs=1 * sdf=50) -> Tăng loss cực mạnh.
        return (probs * sdf).mean()


# ─── Soft CL-Dice Loss (Topology-Preserving Loss) ───────────────────────────

def soft_erode(img):
    """Xói mòn mềm dùng MinPool (phủ định của MaxPool)."""
    if len(img.shape) != 4:
        raise ValueError("Input must be 4D tensor (B, C, H, W)")
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-p1, (1, 3), (1, 1), (0, 1))
    return p2

def soft_dilate(img):
    """Giãn nở mềm dùng MaxPool."""
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

def soft_open(img):
    """Phép mở mềm: Erode sau đó Dilate."""
    return soft_dilate(soft_erode(img))

def soft_skel(img, iters):
    """Trích xuất khung xương mềm lặp lại."""
    img1 = img
    skel = torch.zeros_like(img)
    for _ in range(iters):
        eroded = soft_erode(img1)
        opened = soft_open(eroded)
        skel = skel + F.relu(eroded - opened)
        img1 = eroded
    return skel

class SoftCLDiceLoss(nn.Module):
    """
    Soft Centerline Dice Loss (clDice).
    Đảm bảo tính liên tục của các cấu trúc dạng ống (mạch máu).
    """
    def __init__(self, iters: int = 3, smooth: float = 1e-5):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Trích xuất khung xương (Skeletonization)
        t_skel = soft_skel(targets, self.iters)
        p_skel = soft_skel(probs, self.iters)
        
        # Topology Precision & Sensitivity
        t_prec = ( (p_skel * targets).sum(dim=(1, 2, 3)) + self.smooth ) / \
                 ( p_skel.sum(dim=(1, 2, 3)) + self.smooth )
        
        t_sens = ( (t_skel * probs).sum(dim=(1, 2, 3)) + self.smooth ) / \
                 ( t_skel.sum(dim=(1, 2, 3)) + self.smooth )
        
        cl_dice = (2.0 * t_prec * t_sens) / (t_prec + t_sens + self.smooth)
        return 1.0 - cl_dice.mean()


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
        
        # [NEW] Hausdorff/SDF Boundary Loss cho Lesion
        self.w_lesion_hd = lesion_cfg.get("hd_weight", 0.0)
        if self.w_lesion_hd > 0:
            self.hd_loss_fn = SDFBoundaryLoss()

        # 2. LVO Task — Dùng Modified Focal Loss cho Heatmap Regression
        lvo_cfg = loss_cfg["lvo"]
        self.lvo_main_loss = ModifiedFocalLoss(
            alpha=lvo_cfg.get("mfl_alpha", 2.0),
            beta=lvo_cfg.get("mfl_beta", 4.0),
        )
        self.w_lvo = lvo_cfg["weight"]
        self.w_lvo_boundary = 0.0  # Boundary Loss không phù hợp với Heatmap

        # 3. CoW Task
        cow_cfg = loss_cfg["cow"]
        self.cow_type = cow_cfg.get("type", "tversky")
        self.cow_main_loss = TverskyLoss(alpha=cow_cfg["alpha"], beta=cow_cfg["beta"])
        
        if self.cow_type == "cl_tversky":
            self.cl_loss = SoftCLDiceLoss(iters=cow_cfg.get("iters", 3))
            self.cl_weight = cow_cfg.get("cl_weight", 0.5)
            
        self.w_cow = cow_cfg["weight"]

        # Tham số Uncertainty Weighting (Learnable)
        self.log_vars = nn.ParameterDict({
            "lesion": nn.Parameter(torch.tensor(0.0)),
            "lvo":    nn.Parameter(torch.tensor(0.0)),
            "cow":    nn.Parameter(torch.tensor(0.0)),
        })

        # Ngưỡng chặn Sigma tối thiểu cho từng task (để tránh hố đen tham số)
        u_cfg = config.get("uncertainty", {})
        self.s_min_lesion = math.log(u_cfg.get("s_min_lesion", 0.6)**2)
        self.s_min_lvo    = math.log(u_cfg.get("s_min_lvo", 0.6)**2)
        self.s_min_cow    = math.log(u_cfg.get("s_min_cow", 0.6)**2)

    def forward(self, preds: dict, targets: torch.Tensor) -> dict:
        # 1. Tính toán Main Loss (256x256)
        # Lesion: Kết hợp Tversky (Diện tích) và HD/SDF (Ranh giới)
        l_lesion_main = self.lesion_main_loss(preds["lesion"], targets[:, 0:1])
        
        if hasattr(self, "hd_loss_fn") and targets.shape[1] > 3:
            # SDF nằm ở kênh thứ 4 của targets (index 3)
            l_hd = self.hd_loss_fn(preds["lesion"], targets[:, 3:4])
            l_lesion = (1.0 - self.w_lesion_hd) * l_lesion_main + self.w_lesion_hd * l_hd
        else:
            l_lesion = l_lesion_main

        # LVO dùng ModifiedFocalLoss với Heatmap GT — không dùng Boundary Loss
        l_lvo = self.lvo_main_loss(preds["lvo"], targets[:, 1:2])

        # 3. CoW: Kết hợp Tversky (Diện tích) và Soft-CLDice (Thông suốt)
        l_cow_geom = self.cow_main_loss(preds["cow"], targets[:, 2:3])
        
        if hasattr(self, "cl_loss"):
            l_cl = self.cl_loss(preds["cow"], targets[:, 2:3])
            l_cow = (1.0 - self.cl_weight) * l_cow_geom + self.cl_weight * l_cl
        else:
            l_cow = l_cow_geom

        # Áp dụng Uncertainty Weighting (Kendall et al.) với ngưỡng chặn riêng biệt
        s_lesion = torch.clamp(self.log_vars["lesion"], min=self.s_min_lesion, max=10.0)
        s_lvo    = torch.clamp(self.log_vars["lvo"], min=self.s_min_lvo, max=10.0)
        s_cow    = torch.clamp(self.log_vars["cow"], min=self.s_min_cow, max=10.0)

        # L_total = sum( exp(-s) * L + s )
        main_lesion_weighted = torch.exp(-s_lesion) * l_lesion + s_lesion
        main_lvo_weighted    = torch.exp(-s_lvo)    * l_lvo    + s_lvo
        main_cow_weighted    = torch.exp(-s_cow)    * l_cow    + s_cow

        main_loss = main_lesion_weighted + main_lvo_weighted + main_cow_weighted

        # 2. Tính toán Auxiliary Losses (MDS)
        # aux_masks: [mask_32, mask_64, mask_128, mask_256]
        aux_loss = 0.0
        # Trọng số tăng dần, giảm áp lực ở tầng quá sâu để tránh nhiễu
        aux_weights = [0.05, 0.075, 0.125, 0.25] 
        
        if "aux_masks" in preds and preds["aux_masks"] is not None:
            for i, aux_pred in enumerate(preds["aux_masks"]):
                if aux_pred is None or i >= len(aux_weights): continue
                
                h, w = aux_pred.shape[2], aux_pred.shape[3]

                # [QUAN TRỌNG] Tách biệt logic resize cho từng loại nhãn:
                #
                # LVO (Heatmap):   Dùng adaptive_max_pool2d để BẢO TOÀN ĐỈNH (peak=1.0).
                #   → Nếu dùng 'nearest'/'bilinear', điểm 1.0 đơn lẻ có thể bị mất hoàn toàn.
                #
                # Lesion & CoW (Binary Mask): Dùng interpolate(mode='nearest') để GIỮ NGUYÊN ĐỘ MẢNH.
                #   → Nếu dùng max_pool, nhãn CoW (mạch 1-2 pixel) bị PHÌNH to gấp nhiều lần,
                #     tạo mâu thuẫn tín hiệu giữa Aux Loss (CoW to) và Main Loss (CoW mảnh).
                target_lvo     = F.adaptive_max_pool2d(targets[:, 1:2], output_size=(h, w))
                target_lesion  = F.interpolate(targets[:, 0:1].float(), size=(h, w), mode='nearest')
                target_cow     = F.interpolate(targets[:, 2:3].float(), size=(h, w), mode='nearest')

                # --- Selective Supervision Logic ---
                l_lesion_aux = self.lesion_main_loss(aux_pred[:, 0:1], target_lesion)
                l_cow_aux    = self.cow_main_loss(aux_pred[:, 2:3], target_cow)
                
                # Chỉ tính LVO ở tầng 128 (i=2) và 256 (i=3)
                # Lý do: Ở 32 và 64, điểm LVO bị biến mất do nén ảnh, gây nhiễu gradient.
                if i >= 2:
                    l_lvo_aux = self.lvo_main_loss(aux_pred[:, 1:2], target_lvo)
                    l_aux = self.w_lesion * l_lesion_aux + self.w_lvo * l_lvo_aux + self.w_cow * l_cow_aux
                else:
                    # Ở tầng sâu, chỉ tập trung vào Lesion và CoW (re-scale trọng số)
                    w_sum = self.w_lesion + self.w_cow
                    l_aux = (self.w_lesion / w_sum) * l_lesion_aux + (self.w_cow / w_sum) * l_cow_aux
                
                aux_loss += aux_weights[i] * l_aux

        # 3. Tổng hợp
        total = main_loss + aux_loss
        total = torch.nan_to_num(total, nan=1.0, posinf=1.0, neginf=1.0)

        # Tính Sigma (Uncertainty) để log ra màn hình: sigma = exp(s/2)
        sigma_lesion = torch.exp(s_lesion / 2.0).item()
        sigma_lvo    = torch.exp(s_lvo / 2.0).item()
        sigma_cow    = torch.exp(s_cow / 2.0).item()

        # Trích xuất các thành phần chi tiết để log (dùng .item() để tránh giữ graph)
        # Nếu không có HD/CL thì mặc định là 0
        l_hd_val = l_hd.item() if 'l_hd' in locals() else 0.0
        l_cl_val = l_cl.item() if 'l_cl' in locals() else 0.0

        return {
            "total":  total,
            "main":   main_loss,
            "aux":    aux_loss,
            "lesion": l_lesion,
            "lvo":    l_lvo,
            "cow":    l_cow,
            
            # Chi tiết để theo dõi (Log-only)
            "l_L_tv": l_lesion_main.item(),
            "l_L_hd": l_hd_val,
            "l_C_tv": l_cow_geom.item(),
            "l_C_cl": l_cl_val,
            
            "sigma_lesion": sigma_lesion,
            "sigma_lvo": sigma_lvo,
            "sigma_cow": sigma_cow
        }
