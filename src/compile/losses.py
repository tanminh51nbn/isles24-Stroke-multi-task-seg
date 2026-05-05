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
import torch.distributed as dist
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
    Hệ thống Loss Đa nhiệm Adaptive Competition (DWA+).
    Kết hợp giữa Uncertainty Weighting (Sigma) và Thi đua động (DWA).
    """
    def __init__(self, config: dict):
        super().__init__()
        # 1. Khởi tạo các hàm loss thành phần
        self.lesion_main_loss = FocalTverskyLoss(
            alpha=config["loss"].get("lesion_alpha", 0.7),
            beta=config["loss"].get("lesion_beta", 0.3),
            gamma=config["loss"].get("lesion_gamma", 2.0)
        )
        self.lvo_loss_fn = CurriculumLVOLoss(config)
        self.cow_main_loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=2.0)
        
        # 2. Uncertainty Weighting (Sigma)
        self.log_vars = nn.ParameterDict({
            'lesion': nn.Parameter(torch.tensor(0.5)),
            'lvo': nn.Parameter(torch.tensor(0.5)),
            'cow': nn.Parameter(torch.tensor(0.5))
        })
        
        # 3. Cấu hình DWA+ (Thi đua)
        self.register_buffer('prev_loss', torch.ones(3))      # Loss epoch (t-1)
        self.register_buffer('prev_loss_2', torch.ones(3))    # Loss epoch (t-2)
        self.register_buffer('current_weights', torch.ones(3)) # Trọng số đang áp dụng
        
        self.temp = config["loss"].get("dwa_temperature", 2.0) # Độ gắt của cuộc thi
        self.s_min_lvo = config["loss"].get("s_min_lvo", 0.1)
        self.s_min_lesion = config["loss"].get("s_min_lesion", 0.4)
        self.s_min_cow = config["loss"].get("s_min_cow", 0.6)

        # 4. Lưu trữ tạm thời cho Epoch hiện tại
        self.running_loss = [0.0, 0.0, 0.0]
        self.running_counts = 0

    def update_epoch_stats(self):
        """
        Đồng bộ hóa Loss giữa các GPU (DDP) và cập nhật lịch sử.
        """
        if self.running_counts > 0:
            avg_loss = torch.tensor([l / self.running_counts for l in self.running_loss], 
                                  device=self.prev_loss.device)
            
            # ĐỒNG BỘ HÓA DDP: Cộng tổng avg_loss từ tất cả các GPU
            if dist.is_initialized():
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss / dist.get_world_size()
            
            # Đẩy lịch sử
            self.prev_loss_2.copy_(self.prev_loss)
            self.prev_loss.copy_(avg_loss)
            
            # Reset
            self.running_loss = [0.0, 0.0, 0.0]
            self.running_counts = 0

    def _calculate_dwa(self, epoch):
        if epoch < 2:
            return torch.ones(3, device=self.prev_loss.device)
        
        # Thêm epsilon 1e-8 để tránh chia cho 0
        r = self.prev_loss / (self.prev_loss_2 + 1e-8)
        
        # Ổn định số học cho Softmax (r - r.max)
        r_stable = r - torch.max(r)
        exp_r = torch.exp(r_stable / self.temp)
        w = (exp_r / (exp_r.sum() + 1e-8)) * 3.0
        return w

    def forward(self, preds: dict, targets: torch.Tensor, epoch: int, **kwargs) -> dict:
        # --- PHASE-BASED POLICY (Manual Override) ---
        p_lesion, p_lvo, p_cow = 1.0, 1.0, 1.0
        
        if epoch < 16:
            p_cow, p_lvo, p_lesion = 3.0, 0.1, 0.1
        elif epoch < 40:
            # Sử dụng DWA thi đua
            dwa_w = self._calculate_dwa(epoch).to(targets.device)
            p_lesion, p_lvo, p_cow = dwa_w[0], dwa_w[1], dwa_w[2]
        else:
            # Mệnh lệnh hành chính
            p_lvo, p_lesion, p_cow = 3.0, 2.0, 0.2

        # 1. Tính toán Losses cơ bản
        l_lesion_main = self.lesion_main_loss(preds['lesion'], targets[:, 0:1])
        l_lvo_base, lvo_scale = self.lvo_loss_fn(preds['lvo'], targets[:, 1:2], epoch)
        l_lvo_main = l_lvo_base * lvo_scale
        l_cow_main = self.cow_main_loss(preds['cow'], targets[:, 2:3])

        # 2. Uncertainty Weighting (Sigma)
        s_lesion = torch.clamp(self.log_vars['lesion'], min=self.s_min_lesion, max=10.0)
        s_lvo = torch.clamp(self.log_vars['lvo'], min=self.s_min_lvo, max=10.0)
        s_cow = torch.clamp(self.log_vars['cow'], min=self.s_min_cow, max=10.0)

        # 3. Tổng hợp Main Loss với Trọng số Thi đua
        loss_lesion = (torch.exp(-s_lesion) * l_lesion_main + s_lesion) * p_lesion
        loss_lvo    = (torch.exp(-s_lvo) * l_lvo_main + s_lvo) * p_lvo
        loss_cow    = (torch.exp(-s_cow) * l_cow_main + s_cow) * p_cow
        
        main_loss = loss_lesion + loss_lvo + loss_cow

        # 4. Cập nhật thống kê để DWA học vào epoch sau
        with torch.no_grad():
            self.running_loss[0] += l_lesion_main.item()
            self.running_loss[1] += l_lvo_main.item()
            self.running_loss[2] += l_cow_main.item()
            self.running_counts += 1

        # 5. Deep Supervision (AUX) - Áp dụng hỏa lực thi đua (p_task)
        aux_loss = 0.0
        if "aux_masks" in preds and preds["aux_masks"] is not None:
            aux_weights = [0.05, 0.075, 0.125, 0.25]
            aux_list = preds["aux_masks"] # List of 4 tensors, each (B, 3, H, W)
            
            for i, aux_pred_3ch in enumerate(aux_list):
                if aux_pred_3ch is None or i >= len(aux_weights):
                    continue
                
                h, w = aux_pred_3ch.shape[2], aux_pred_3ch.shape[3]
                
                # Tách 3 kênh: 0=Lesion, 1=LVO, 2=CoW
                aux_lesion = aux_pred_3ch[:, 0:1]
                aux_lvo    = aux_pred_3ch[:, 1:2]
                aux_cow    = aux_pred_3ch[:, 2:3]
                
                # 1. Lesion Aux Loss
                t_lesion = F.interpolate(targets[:, 0:1].float(), (h, w), mode='nearest')
                l_a_lesion = self.lesion_main_loss(aux_lesion, t_lesion) * p_lesion
                
                # 2. LVO Aux Loss
                t_lvo = F.adaptive_max_pool2d(targets[:, 1:2], (h, w))
                l_a_lvo_base, _ = self.lvo_loss_fn(aux_lvo, t_lvo, epoch)
                l_a_lvo = l_a_lvo_base * p_lvo
                
                # 3. CoW Aux Loss
                t_cow = F.interpolate(targets[:, 2:3].float(), (h, w), mode='nearest')
                l_a_cow = self.cow_main_loss(aux_cow, t_cow) * p_cow
                
                aux_loss += aux_weights[i] * (l_a_lesion + l_a_lvo + l_a_cow)

        # 6. Tổng hợp kết quả
        total = main_loss + aux_loss
        
        return {
            'total': total,
            'main': main_loss,
            'aux': aux_loss,
            'l_lesion': loss_lesion.item(),
            'l_lvo': loss_lvo.item(),
            'l_cow': loss_cow.item(),
            'p_lvo': p_lvo if isinstance(p_lvo, float) else p_lvo.item(),
            'sigma_lvo': torch.exp(s_lvo * 0.5).item()
        }
