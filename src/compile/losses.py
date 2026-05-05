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
from typing import Tuple


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
    Hỗ trợ Gaussian Blurring on-the-fly để thực hiện Curriculum Learning.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    def _apply_gaussian_blur(self, mask: torch.Tensor, sigma: float) -> torch.Tensor:
        """Áp dụng Gaussian Blur on-the-fly trên GPU."""
        if sigma <= 0:
            return mask
        
        kernel_size = int(sigma * 4)
        if kernel_size % 2 == 0: kernel_size += 1
        
        # Tạo Gaussian kernel
        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-x.pow(2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_2d = kernel_1d.view(1, 1, -1, 1) * kernel_1d.view(1, 1, 1, -1)
        kernel_2d = kernel_2d.to(mask.device)
        
        # Thêm padding để giữ nguyên size
        padding = kernel_size // 2
        blurred = F.conv2d(mask, kernel_2d, padding=padding)
        
        # Normalize peak = 1.0 (như make_lvo_heatmap trong dataset.py)
        # Thực hiện per-instance (batch-wise)
        peaks = blurred.view(blurred.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        return blurred / peaks.clamp(min=1e-6)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sigma: float = 0.0) -> torch.Tensor:
        """
        Args:
            logits:  Raw logits (B, 1, H, W)
            targets: Binary mask (B, 1, H, W)
            sigma:   Độ nhòe của heatmap.
        """
        logits = logits.float()
        
        # Tạo Heatmap từ Binary Mask on-the-fly
        heatmap_gt = self._apply_gaussian_blur(targets, sigma) if sigma > 0 else targets
        heatmap_gt = heatmap_gt.float()

        pred = torch.sigmoid(logits)
        pred = pred.clamp(min=self.eps, max=1.0 - self.eps)

        pos_mask = (heatmap_gt == 1.0).float()
        neg_mask = 1.0 - pos_mask

        pos_loss = -pos_mask * torch.pow(1.0 - pred, self.alpha) * torch.log(pred)
        neg_loss = -neg_mask * torch.pow(1.0 - heatmap_gt, self.beta) * \
                   torch.pow(pred, self.alpha) * torch.log(1.0 - pred)

        pos_loss = pos_loss.sum(dim=(1, 2, 3))
        neg_loss = neg_loss.sum(dim=(1, 2, 3))
        
        num_pos = pos_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        return ((pos_loss + neg_loss) / num_pos).mean()


# ─── Curriculum LVO Loss ─────────────────────────────────────────────────────

class CurriculumLVOLoss(nn.Module):
    """
    Điều phối lộ trình học tập (Curriculum) cho task LVO.
    """
    def __init__(self, config: dict):
        super().__init__()
        lvo_cfg = config["loss"]["lvo"]
        self.mfl = ModifiedFocalLoss(
            alpha=lvo_cfg.get("mfl_alpha", 2.0),
            beta=lvo_cfg.get("mfl_beta", 4.0)
        )
        self.ftl = FocalTverskyLoss(
            alpha=0.2, beta=0.8, gamma=2.0 # Mặc định cho giai đoạn sau
        )
        
        # Cấu hình lộ trình từ train.yaml
        curr_cfg = config["training"].get("curriculum_learning", {})
        self.enabled = curr_cfg.get("enabled", False)
        
        # Ngưỡng chuyển giai đoạn
        self.phase1_end = 6  # Epoch 0-5
        self.phase2_end = 16 # Epoch 6-15

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch: int) -> Tuple[torch.Tensor, float]:
        """
        Trả về: (loss_value, weight_scale)
        """
        if not self.enabled:
            return self.mfl(logits, targets, sigma=4.0), 1.0

        # Giai đoạn 1: Warmup (Soft Heatmap) - Epoch 0-5
        if epoch < 6:
            loss = self.mfl(logits, targets, sigma=10.0)
            return loss, 1.0 # Trọng số cơ bản

        # Giai đoạn 2: Ramp-up (Medium Heatmap) - Epoch 6-15
        elif epoch < 16:
            loss = self.mfl(logits, targets, sigma=5.0)
            return loss, 1.5 # Bắt đầu ưu tiên LVO

        # Giai đoạn 3: Hard Mining (Sharp Heatmap) - Epoch 16-39
        # [NEW] Kéo dài giai đoạn này để mô hình học kỹ không gian trước khi sang Binary
        elif epoch < 40:
            loss = self.mfl(logits, targets, sigma=3.0)
            return loss, 2.5 # Ưu tiên mạnh (x2.5)

        # Giai đoạn 4: Final Push (Strict Binary) - Epoch 40+
        else:
            # Dùng Focal Tversky để tập trung vào precision và bứt phá Recall
            loss = self.ftl(logits, targets) 
            return loss, 5.0 # TẤT TAY (x5.0)

    def get_mds_boost(self, epoch: int) -> float:
        """
        [FIX] MDS Boost Schedule để tránh Gradient Shock.
        Khởi động nhẹ nhàng (x1.0) và bứt phá ở giai đoạn cuối (x3.5).
        """
        if epoch < 6:
            return 1.0  # Giai đoạn 1: Warmup
        elif epoch < 16:
            return 2.5  # Giai đoạn 2: Ramp-up
        elif epoch < 40:
            return 5.0  # Giai đoạn 3: Hard Mining
        else:
            return 8.0  # Giai đoạn 4: Final Push (Balanced Aggressive)



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
            sdf:    Bản đồ khoảng cách từ ground truth (B, 1, H, W) trong khoảng [-1, 1]
        """
        probs = torch.sigmoid(logits)
        
        # 1. Phạt lỗi "Dự đoán nhầm ở ngoài" (False Positives)
        # sdf > 0 cho vùng bên ngoài
        fp_loss = probs * torch.clamp(sdf, min=0)
        
        # 2. Phạt lỗi "Bỏ sót ở trong" (False Negatives)
        # sdf < 0 cho vùng bên trong
        fn_loss = (1 - probs) * torch.abs(torch.clamp(sdf, max=0))
        
        # Tổng hợp: Kết quả sẽ luôn >= 0 và <= 1
        return (fp_loss + fn_loss).mean()


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

        # 2. LVO Task — Dùng Curriculum Loss
        self.lvo_loss_fn = CurriculumLVOLoss(config)
        self.w_lvo = loss_cfg["lvo"]["weight"]
        self.w_lvo_boundary = 0.0

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

    def forward(self, preds: dict, targets: torch.Tensor, epoch: int = 0, batch_idx: int = 0) -> dict:
        # 1. Tính toán Main Loss (256x256)
        # Lesion: Kết hợp Tversky (Diện tích) và HD/SDF (Ranh giới)
        l_lesion_main = self.lesion_main_loss(preds["lesion"], targets[:, 0:1])
        
        if hasattr(self, "hd_loss_fn") and targets.shape[1] > 3:
            # SDF nằm ở kênh thứ 4 của targets (index 3)
            l_hd = self.hd_loss_fn(preds["lesion"], targets[:, 3:4])
            l_lesion = (1.0 - self.w_lesion_hd) * l_lesion_main + self.w_lesion_hd * l_hd
        else:
            l_lesion = l_lesion_main

        # LVO: Dùng Curriculum Learning (trả về loss và weight_scale)
        l_lvo_base, lvo_scale = self.lvo_loss_fn(preds["lvo"], targets[:, 1:2], epoch)
        l_lvo = l_lvo_base * lvo_scale

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

        # [CẢI TIẾN] Task Priority Scheduling: Gate MUST learn first
        p_cow, p_lvo, p_lesion = 1.0, 1.0, 1.0
        if epoch < 16:
            p_cow, p_lvo, p_lesion = 3.0, 0.1, 0.1
        elif epoch >= 40:
            p_lvo = 2.0 
            p_lesion = 1.5

        main_lesion_weighted *= p_lesion
        main_lvo_weighted    *= p_lvo
        main_cow_weighted    *= p_cow

        main_loss = main_lesion_weighted + main_lvo_weighted + main_cow_weighted
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
                # LVO (Heatmap/Binary): Dùng adaptive_max_pool2d để BẢO TOÀN ĐỈNH.
                # Lesion & CoW (Binary Mask): Dùng interpolate(mode='nearest') để GIỮ NGUYÊN ĐỘ MẢNH.
                target_lvo     = F.adaptive_max_pool2d(targets[:, 1:2], output_size=(h, w))
                target_lesion  = F.interpolate(targets[:, 0:1].float(), size=(h, w), mode='nearest')
                target_cow     = F.interpolate(targets[:, 2:3].float(), size=(h, w), mode='nearest')

                # --- Selective Supervision Logic ---
                l_lesion_aux = self.lesion_main_loss(aux_pred[:, 0:1], target_lesion)
                l_cow_aux    = self.cow_main_loss(aux_pred[:, 2:3], target_cow)
                
                # [NEW] Sync Priority with Main Loss
                p_cow, p_lvo, p_lesion = 1.0, 1.0, 1.0
                if epoch < 16:
                    p_cow, p_lvo, p_lesion = 3.0, 0.1, 0.1
                elif epoch >= 40:
                    p_lvo = 2.0 
                    p_lesion = 1.5

                if i >= 2:
                    # Chú ý: Ở Aux layer cũng dùng Curriculum
                    l_lvo_aux_base, lvo_aux_scale = self.lvo_loss_fn(aux_pred[:, 1:2], target_lvo, epoch)
                    l_lvo_aux = l_lvo_aux_base * lvo_aux_scale
                    
                    # MDS Balancing: Boost LVO động theo epoch
                    mds_boost = self.lvo_loss_fn.get_mds_boost(epoch)
                    l_aux = (self.w_lesion * l_lesion_aux * p_lesion) + \
                            (self.w_lvo * l_lvo_aux * mds_boost * p_lvo) + \
                            (self.w_cow * l_cow_aux * p_cow)
                else:
                    # Ở tầng sâu, chỉ tập trung vào Lesion và CoW
                    w_sum = self.w_lesion + self.w_cow
                    l_aux = (self.w_lesion / w_sum * l_lesion_aux * p_lesion) + \
                            (self.w_cow / w_sum * l_cow_aux * p_cow)
                
                aux_loss += aux_weights[i] * l_aux

                # [DEBUG] Verify MDS ratio định kỳ ở Epoch 0
                if i == 3 and epoch == 0 and batch_idx % 100 == 0:
                    # In trực tiếp ra console để verify logic cân bằng
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                         print(f"  [MDS_CHECK] B{batch_idx} LVO_Aux: {l_lvo_aux.item():.4f} | Lesion_Aux: {l_lesion_aux.item():.4f} (Ratio: {l_lvo_aux.item()/(l_lesion_aux.item()+1e-6):.2f})")

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
