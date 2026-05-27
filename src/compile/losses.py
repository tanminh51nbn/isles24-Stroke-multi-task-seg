"""
losses.py — Task-specific Loss Functions cho Multi-Task Stroke Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, Optional

# ─── Tversky Loss ─────────────────────────────────────────────────────────────
 
class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0, batch: bool = True, reduction: str = 'mean'):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth
        self.batch  = batch
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.float())
        targets = targets.float()
        if self.batch:
            probs   = probs.contiguous().view(-1)
            targets = targets.contiguous().view(-1)
            TP = (probs * targets).sum()
            FP = (probs * (1 - targets)).sum()
            FN = ((1 - probs) * targets).sum()
            numerator   = TP + self.smooth
            denominator = TP + self.alpha * FP + self.beta * FN + self.smooth
            tversky_index = numerator / denominator.clamp(min=self.smooth)
            return 1.0 - tversky_index
        else:
            probs   = probs.view(probs.size(0), -1)
            targets = targets.view(targets.size(0), -1)
            TP = (probs * targets).sum(dim=1)
            FP = (probs * (1 - targets)).sum(dim=1)
            FN = ((1 - probs) * targets).sum(dim=1)
            numerator   = TP + self.smooth
            denominator = TP + self.alpha * FP + self.beta * FN + self.smooth
            tversky_index = numerator / denominator.clamp(min=self.smooth)
            loss_vec = 1.0 - tversky_index
            if self.reduction == 'mean':
                return loss_vec.mean()
            else:
                return loss_vec

# ─── Focal Tversky Loss ───────────────────────────────────────────────────────

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 2.0, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth
        self.reduction = reduction

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
        error = (1.0 - tversky_index).clamp(min=1e-6, max=1.0)
        loss_vec = torch.pow(error, self.gamma)
        if self.reduction == 'mean':
            return loss_vec.mean()
        else:
            return loss_vec

def apply_gaussian_blur(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0: return mask
    kernel_size = int(sigma * 4)
    if kernel_size % 2 == 0: kernel_size += 1
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-x.pow(2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.view(1, 1, -1, 1) * kernel_1d.view(1, 1, 1, -1)
    kernel_2d = kernel_2d.to(mask.device)
    padding = kernel_size // 2
    blurred = F.conv2d(mask, kernel_2d, padding=padding)
    peaks = blurred.view(blurred.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    blurred = blurred / peaks.clamp(min=1e-6)
    return torch.max(mask, blurred)

# ─── Modified Focal Loss ──────────────────────────────────────────────────────

class ModifiedFocalLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sigma: float = 0.0, debug: bool = False) -> torch.Tensor:
        pred = torch.sigmoid(logits.float()).clamp(min=self.eps, max=1.0 - self.eps)
        
        # 1. Vùng Đỏ (Nhãn gốc) và Vùng Nền (Tất cả còn lại)
        # Sử dụng nguyên gốc targets thay vị trí đỉnh của heatmap
        pos_mask = (targets >= 0.5).float()
        neg_mask = 1.0 - pos_mask
        
        # 2. Bản đồ Không gian (Heatmap lan tỏa)
        heatmap_gt = apply_gaussian_blur(targets, sigma) if sigma > 0 else targets
        
        # 3. Tính Lực Thưởng và Lực Phạt
        pos_loss = -pos_mask * torch.pow(1.0 - pred, self.alpha) * torch.log(pred + self.eps)
        neg_loss = -neg_mask * torch.pow(1.0 - heatmap_gt, self.beta) * torch.pow(pred, self.alpha) * torch.log(1.0 - pred + self.eps)
        
        # 4. Tính toán theo từng lát cắt (dim = (1, 2, 3)) để hỗ trợ slice-level weights
        slice_pos_loss = pos_loss.sum(dim=(1, 2, 3))
        slice_neg_loss = neg_loss.sum(dim=(1, 2, 3))
        
        slice_num_pos = pos_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        slice_num_neg = neg_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        
        # Trừng phạt thiết quân luật cục bộ trên từng lát cắt
        slice_weight_pos = (slice_num_neg / slice_num_pos).clamp(max=200)
        
        slice_loss_pos = slice_pos_loss * slice_weight_pos
        slice_loss = (slice_loss_pos + slice_neg_loss) / (slice_num_neg + slice_num_pos * slice_weight_pos)
        
        slice_loss = torch.nan_to_num(slice_loss, nan=0.0, posinf=100.0, neginf=0.0).clamp(max=100.0)
        
        if debug:
            num_pos_val = int(pos_mask.sum().item())
            print(f"      [MFL_DEBUG] pos_px={num_pos_val} mean_loss={slice_loss.mean().item():.4f}")

        if self.reduction == 'mean':
            return slice_loss.mean()
        else:
            return slice_loss

# ─── LVO Loss ────────────────────────────────────────────────────────────────
# Sử dụng Modified Focal Loss để tập trung vào các điểm tắc mạch nhỏ (Keypoints)

# ─── Boundary Losses ──────────────────────────────────────────────────────────

class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        t_b = targets - (1.0 - F.max_pool2d(1.0 - targets, kernel_size=self.kernel_size, stride=1, padding=self.padding))
        t_b = torch.clamp(t_b, min=0.0)
        p_b = probs - (1.0 - F.max_pool2d(1.0 - probs, kernel_size=self.kernel_size, stride=1, padding=self.padding))
        p_b = torch.clamp(p_b, min=0.0)
        intersection = (p_b * t_b).sum(dim=(1, 2, 3))
        union = p_b.sum(dim=(1, 2, 3)) + t_b.sum(dim=(1, 2, 3))
        return 1.0 - ((2.0 * intersection + 1e-5) / (union + 1e-5)).mean()

class SDFBoundaryLoss(nn.Module):
    def __init__(self, fg_weight: float = 0.1):
        super().__init__()
        self.fg_weight = fg_weight

    def forward(self, logits: torch.Tensor, sdf: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = logits.float()
        sdf = sdf.float()
        if mask is not None:
            mask = mask.float()
        if mask is None:
            # Fallback for compatibility (e.g. tests or older configurations)
            probs = torch.sigmoid(logits)
            fp_loss = probs * torch.clamp(sdf, min=0)
            fn_loss = (1 - probs) * torch.abs(torch.clamp(sdf, max=0))
            return (fp_loss + fn_loss).mean()

        # mask shape: (B, 1, H, W)
        # 1. Slice-Level Gating: Only compute loss on slices that actually contain lesions
        has_lesion = (mask.sum(dim=(1, 2, 3)) > 0)
        if not has_lesion.any():
            return torch.tensor(0.0, device=logits.device)

        logits_pos = logits[has_lesion]
        sdf_pos = sdf[has_lesion]

        probs = torch.sigmoid(logits_pos)

        # 2. Foreground-Background Balancing: compute mean fp_loss and fn_loss separately
        # Background pixels have sdf > 0
        is_bg = sdf_pos > 0
        # Foreground pixels have sdf <= 0
        is_fg = sdf_pos <= 0

        num_bg = is_bg.sum().float()
        num_fg = is_fg.sum().float()

        # Calculate losses
        fp_loss = probs * torch.clamp(sdf_pos, min=0)
        fn_loss = (1 - probs) * torch.abs(torch.clamp(sdf_pos, max=0))

        # Average separately to balance gradients
        fp_mean = fp_loss.sum() / (num_bg + 1e-8)
        fn_mean = fn_loss.sum() / (num_fg + 1e-8)

        return fp_mean + self.fg_weight * fn_mean

# ─── Soft clDice ─────────────────────────────────────────────────────────────

def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-p1, (1, 3), (1, 1), (0, 1))
    return p2

def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iters):
    img1 = img
    skel = torch.zeros_like(img)
    for _ in range(iters):
        eroded = soft_erode(img1)
        opened = soft_open(eroded)
        skel = skel + F.relu(eroded - opened)
        img1 = eroded
    return skel

class SoftCLDiceLoss(nn.Module):
    def __init__(self, iters: int = 3, smooth: float = 1.0):
        super().__init__()
        self.iters, self.smooth = iters, smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        t_s, p_s = soft_skel(targets, self.iters), soft_skel(probs, self.iters)
        t_p = ((p_s * targets).sum() + self.smooth) / (p_s.sum() + self.smooth)
        t_s_ = ((t_s * probs).sum() + self.smooth) / (t_s.sum() + self.smooth)
        return 1.0 - ((2.0 * t_p * t_s_) / (t_p + t_s_ + self.smooth))

# ─── Compound Dice + BCE Loss ──────────────────────────────────────────────────

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.float())
        targets = targets.float()
        
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        alpha_t = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal_loss = alpha_t * torch.pow(1.0 - pt, self.gamma) * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            # Return slice-level vector of shape (B,) by averaging over spatial dimensions
            return focal_loss.view(focal_loss.size(0), -1).mean(dim=1)

class CompoundDiceBCELoss(nn.Module):
    """
    nnU-Net compound loss cải tiến: 0.5 * Dice (Tversky) + 0.5 * BCEWithLogitsLoss (hoặc FocalLoss).
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0, pos_weight: float = 1.0, batch: bool = False, use_focal: bool = False, focal_alpha: float = 0.25, focal_gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.dice = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth, batch=batch, reduction=reduction)
        self.use_focal = use_focal
        self.reduction = reduction
        if use_focal:
            self.focal = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)
        else:
            self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        
        dice_loss = self.dice(logits, targets)
        if self.use_focal:
            bce_loss = self.focal(logits, targets)
        else:
            if self.reduction == 'none':
                # Compute per-pixel BCE, then average over spatial dimensions (1, 2, 3) to get (B,)
                bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
                bce_loss = bce_loss.view(bce_loss.size(0), -1).mean(dim=1)
            else:
                bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        
        return 0.5 * dice_loss + 0.5 * bce_loss


# ─── Multi-Task Loss (The Core) ───────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        l_cfg  = config["loss"]
        t_cfg  = config.get("training", {})
        self.init_w = t_cfg.get("initial_weights", {"lesion": 1.0, "lvo": 1.0, "cow": 1.0})

        # 1. Lesion Task — ModifiedFocalLoss với Gaussian Heatmap Curriculum
        # Giống LVO nhưng tuned cho region: alpha nhỏ hơn, beta nhỏ hơn, sigma sàn cao hơn.
        self.lesion_main_loss = ModifiedFocalLoss(
            alpha=l_cfg["lesion"].get("mfl_alpha", 1.5),
            beta=l_cfg["lesion"].get("mfl_beta",  2.0),
            reduction='none'
        )
        # Sigma curriculum: bắt đầu mờ (dạy vị trí), dần sharp (dạy biên)
        self.lesion_sigma_init  = l_cfg["lesion"].get("sigma_init",  9.0)
        self.lesion_sigma_floor = l_cfg["lesion"].get("sigma_floor", 4.0)
        self.lesion_sigma_decay = l_cfg["lesion"].get("sigma_decay", 0.97)
        self.lesion_slice_pos_w = l_cfg["lesion"].get("slice_pos_weight", 3.0)
        l_l_cls_pos_w = l_cfg["lesion"].get("cls_pos_weight", 2.0)
        self.lesion_cls_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([l_l_cls_pos_w])
        )
        self.lesion_cls_w = l_cfg["lesion"].get("cls_weight", 0.25)

        # Soft Dice Loss (TverskyLoss with alpha=0.5, beta=0.5) to optimize volumetric overlap directly
        self.lesion_dice_loss_fn = TverskyLoss(
            alpha=0.5,
            beta=0.5,
            smooth=1.0,
            batch=False,
            reduction='none'
        )
        self.lesion_dice_w = l_cfg["lesion"].get("dice_weight", 0.5)

        # 2. LVO Task
        l_v_cfg = l_cfg.get("lvo", {})
        lvo_type = l_v_cfg.get("type", "modified_focal")
        if lvo_type == "focal_tversky":
            self.lvo_loss_fn = FocalTverskyLoss(
                alpha=l_v_cfg.get("alpha", 0.7),
                beta=l_v_cfg.get("beta", 0.3),
                gamma=l_v_cfg.get("gamma", 2.0),
                reduction='none'
            )
        else:
            self.lvo_loss_fn = ModifiedFocalLoss(
                alpha=l_v_cfg.get("mfl_alpha", 2.0),
                beta=l_v_cfg.get("mfl_beta", 4.0),
                reduction='none'
            )
        # [T2.1] LVO Binary Classification Loss
        # [FIX] Hạ pos_weight từ 5.0 xuống 1.5 để dập tắt ảo giác (giảm >1200 FPs)
        lvo_cls_pos_w = l_v_cfg.get("cls_pos_weight", l_v_cfg.get("lvo_cls_pos_weight", 1.5))
        self.lvo_cls_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([lvo_cls_pos_w])
        )
        self.lvo_cls_w = l_v_cfg.get("cls_weight", l_v_cfg.get("lvo_cls_weight", 0.3))  # Tỷ lệ cls trong tổng LVO loss
        
        # Sigma curriculum và slice weights cho LVO:
        self.lvo_sigma_init  = l_v_cfg.get("sigma_init",  7.5)
        self.lvo_sigma_floor = l_v_cfg.get("sigma_floor", 2.0)
        self.lvo_sigma_decay = l_v_cfg.get("sigma_decay", 0.96)
        self.lvo_slice_pos_w = l_v_cfg.get("slice_pos_weight", 3.0)

        # 3. CoW Task
        # [FIX] Dùng TverskyLoss (linear gradient) thay FocalTverskyLoss (gamma=2, gradient²)
        # Tại Dice=0.835: gradient FTL ≈ 0.33x, gradient TL = 1.0x → mạnh hơn 3× để thoát local minimum
        self.cow_main_loss = TverskyLoss(
            alpha=l_cfg["cow"].get("alpha", 0.5),
            beta=l_cfg["cow"].get("beta", 0.5),
            batch=True
        )
        self.cow_cl_loss = SoftCLDiceLoss(iters=l_cfg["cow"].get("iters", 3))
        self.cow_cl_w = l_cfg["cow"].get("cl_weight", 0.0)

        # ── Performance Gap Weighting (PGW) ────────────────────────────────────
        # Điều chỉnh trọng số dựa trên khoảng cách đến target metric,
        # không phải tốc độ thay đổi loss — tránh failure mode "đầu hàng task khó".
        pgw_cfg = t_cfg.get("pgw", {})
        perf_tgt = pgw_cfg.get("performance_targets", {})
        self.perf_targets = torch.tensor([
            perf_tgt.get("lesion", 0.85),
            perf_tgt.get("lvo",    0.50),
            perf_tgt.get("cow",    0.90),
        ])
        self.pgw_temperature  = pgw_cfg.get("temperature", 0.5)
        self.pgw_momentum     = pgw_cfg.get("momentum",    0.3)
        w_min_cfg = pgw_cfg.get("w_min", 0.1)
        if isinstance(w_min_cfg, dict):
            self.register_buffer('pgw_w_min', torch.tensor([
                w_min_cfg.get("lesion", 0.1),
                w_min_cfg.get("lvo",    0.1),
                w_min_cfg.get("cow",    0.1),
            ], dtype=torch.float32))
        elif isinstance(w_min_cfg, (list, tuple)):
            self.register_buffer('pgw_w_min', torch.tensor(w_min_cfg, dtype=torch.float32))
        else:
            self.register_buffer('pgw_w_min', torch.tensor([w_min_cfg, w_min_cfg, w_min_cfg], dtype=torch.float32))
        self.pgw_start_epoch  = pgw_cfg.get("start_epoch", 5)
        self.n_tasks          = 3
        # Per-task weight ceiling: ngăn task collapse chiếm đoạt toàn bộ budget
        w_max_cfg = pgw_cfg.get("w_max", {})
        self.pgw_w_max = torch.tensor([
            w_max_cfg.get("lesion", 3.0),
            w_max_cfg.get("lvo",    3.0),
            w_max_cfg.get("cow",    3.0),
        ])

        # current_weights: khởi tạo từ initial_weights trong config
        self.register_buffer('current_weights', torch.tensor([
            self.init_w['lesion'],
            self.init_w['lvo'],
            self.init_w['cow'],
        ]))
        # Buffer lưu epoch cuối cùng được cập nhật (để biết PGW đã kích hoạt chưa)
        self.register_buffer('_pgw_epoch', torch.tensor(-1, dtype=torch.long))

    def update_weights_from_metrics(self, val_metrics: dict, epoch: int):
        """Cập nhật current_weights theo Performance Gap Weighting (PGW).

        Gọi 1 lần/epoch từ trainer.py (tất cả các rank) SAU KHI validate() xong.
        current_weights được cập nhật in-place → forward() chỉ đọc, không tính lại.
        Không còn Stall Detection — Lesion luôn tham gia PGW bình thường.
        """
        if epoch < self.pgw_start_epoch:
            return  # Giữ initial_weights trong giai đoạn warmup

        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            current = torch.tensor([
                # Dùng dice_lesion_pos (chỉ trên Lesion-positive slice) để PGW thấy đúng gap thực sự
                val_metrics.get("dice_lesion_pos", val_metrics.get("dice_lesion", 0.0)),
                val_metrics.get("f1_lvo",    0.0) / 100.0,  # F1 → [0,1]
                val_metrics.get("dice_cow",  0.0),
            ], device=self.current_weights.device)

            target = self.perf_targets.to(current.device)
            gap = torch.clamp(target - current, min=0.0)  # Chỉ tính gap dương (chưa đạt target)

            if gap.sum() < 1e-6:
                # Tất cả task đã vượt target → dùng lại initial_weights
                init = torch.tensor(
                    [self.init_w['lesion'], self.init_w['lvo'], self.init_w['cow']],
                    device=self.current_weights.device
                )
                self.current_weights.copy_(init)
                self._pgw_epoch.fill_(epoch)
                print(f"    [PGW] Tất cả task đạt target -> reset về initial_weights")
            else:
                # Tách task ĐÃ đạt target (gap≈0) và CHƯA đạt.
                # Task đã đạt → giữ cố định ở w_min, KHÔNG tham gia softmax.
                active_mask = gap > 1e-3
                n_active   = int(active_mask.sum().item())

                if n_active == 0:
                    init = torch.tensor(
                        [self.init_w['lesion'], self.init_w['lvo'], self.init_w['cow']],
                        device=self.current_weights.device
                    )
                    self.current_weights.copy_(init)
                    self._pgw_epoch.fill_(epoch)
                else:
                    # Softmax chỉ trên active tasks; budget = N - sum(w_min for inactive tasks)
                    inactive_w_min_sum = (self.pgw_w_min * (~active_mask).float()).sum()
                    weight_budget = self.n_tasks - inactive_w_min_sum
                    active_gap    = gap * active_mask.float()
                    # Subtract max before exp (Softmax stability trick)
                    max_gap       = torch.max(active_gap) if n_active > 0 else 0.0
                    exp_gap       = torch.exp((active_gap - max_gap) / self.pgw_temperature) * active_mask.float()
                    raw_w = torch.where(
                        active_mask,
                        (exp_gap / (exp_gap.sum() + 1e-8)) * weight_budget,
                        self.pgw_w_min
                    )

                    # Momentum + Phân bổ lại phần thừa (Projected Simplex với Box Constraints)
                    blended_w = self.pgw_momentum * self.current_weights + (1.0 - self.pgw_momentum) * raw_w

                    w = blended_w.clone()
                    w_min = self.pgw_w_min.to(w.device)
                    w_max = self.pgw_w_max.to(w.device)

                    # Phân bổ phần thừa tuần tự (tối đa N bước cho N tasks)
                    for _ in range(self.n_tasks):
                        w = torch.clamp(w, min=w_min, max=w_max)
                        not_at_bounds = (w > w_min) & (w < w_max)
                        if not not_at_bounds.any():
                            break
                        excess = self.n_tasks - w.sum()
                        if abs(excess.item()) < 1e-4:
                            break
                        n_free = not_at_bounds.sum().float()
                        w[not_at_bounds] += excess / n_free

                    final_w = torch.clamp(w, min=w_min, max=w_max)
                    self.current_weights.copy_(final_w)
                    self._pgw_epoch.fill_(epoch)

                    fixed_tag = "".join([
                        "L" if not active_mask[0] else "",
                        "V" if not active_mask[1] else "",
                        "C" if not active_mask[2] else "",
                    ]) or "none"
                    print(f"    [PGW]: Gap L={gap[0]:.3f} V={gap[1]:.3f} C={gap[2]:.3f} "
                          f"-> Weights: L={final_w[0]:.2f} V={final_w[1]:.2f} C={final_w[2]:.2f} "
                          f"(fixed: {fixed_tag})")

        if dist.is_initialized():
            dist.broadcast(self.current_weights, src=0)
            dist.broadcast(self._pgw_epoch, src=0)

    def forward(self, preds: dict, targets: torch.Tensor, epoch: int, **kwargs) -> dict:
        cur_ep = int(epoch)
        # Chỉ ĐỌC current_weights — được PGW cập nhật 1 lần/epoch từ update_weights_from_metrics().
        # Trước pgw_start_epoch: current_weights = initial_weights (set trong __init__).
        w = self.current_weights.to(targets.device)
        p_l, p_v, p_c = w[0], w[1], w[2]

        dynamic_sigma  = max(self.lvo_sigma_floor, self.lvo_sigma_init * (self.lvo_sigma_decay ** cur_ep))
        lesion_sigma   = max(self.lesion_sigma_floor, self.lesion_sigma_init * (self.lesion_sigma_decay ** cur_ep))
        _log = kwargs.get('batch_idx', 0) == 0 and (not dist.is_initialized() or dist.get_rank() == 0)
        if _log:
            pgw_active = self._pgw_epoch.item() >= 0
            tag = "[PGW]" if pgw_active else "[INIT]"
            print(f"    {tag}: Weights [Lesion={p_l:.2f}, LVO={p_v:.2f}, CoW={p_c:.2f}] "
                  f"| LVO_Sigma={dynamic_sigma:.2f} | Les_Sigma={lesion_sigma:.2f}")

        # 1. Main Losses
        _debug = (kwargs.get('batch_idx', -1) == 0) and (not dist.is_initialized() or dist.get_rank() == 0)
        # Lesion: Hybrid Loss (ModifiedFocalLoss + Soft Dice Loss)
        l_l_m_focal = self.lesion_main_loss(preds['lesion'], targets[:, 0:1], sigma=lesion_sigma)
        l_l_m_dice = self.lesion_dice_loss_fn(preds['lesion'], targets[:, 0:1])
        l_l_m = (1.0 - self.lesion_dice_w) * l_l_m_focal + self.lesion_dice_w * l_l_m_dice

        if isinstance(self.lvo_loss_fn, ModifiedFocalLoss):
            l_v_m = self.lvo_loss_fn(preds['lvo'], targets[:, 1:2], sigma=dynamic_sigma, debug=_debug)
        else:
            # Focal Tversky Loss on Heatmap
            heatmap_gt = apply_gaussian_blur(targets[:, 1:2], dynamic_sigma) if dynamic_sigma > 0 else targets[:, 1:2]
            l_v_m = self.lvo_loss_fn(preds['lvo'], heatmap_gt)
            if _debug:
                num_pos = (targets[:, 1:2] > 0.5).sum().item()
                print(f"      [LVO_DEBUG] PosPixels: {int(num_pos)} | FTL_Loss: {l_v_m.mean().item():.4f}")
        l_c_m = self.cow_main_loss(preds['cow'], targets[:, 2:3])

        # Slice indicators (1.0 if there is any GT pixel, else 0.0)
        has_lesion = (targets[:, 0:1].amax(dim=(1, 2, 3), keepdim=False) > 0).float()  # (B,)
        has_lvo = (targets[:, 1:2].amax(dim=(1, 2, 3), keepdim=False) > 0).float()      # (B,)

        # [T2.1] LVO Binary Classification Loss (per-slice vector)
        lvo_cls_logit = preds.get('lvo_cls', None)
        if lvo_cls_logit is not None:
            lvo_cls_logit_flat = lvo_cls_logit.view(-1)  # (B,)
            pos_w = self.lvo_cls_loss_fn.pos_weight.to(targets.device)
            l_v_cls = nn.functional.binary_cross_entropy_with_logits(
                lvo_cls_logit_flat, has_lvo,
                pos_weight=pos_w,
                reduction='none'
            )
            l_v_m = (1.0 - self.lvo_cls_w) * l_v_m + self.lvo_cls_w * l_v_cls

        # 2. Additional Task-specific Losses (Boundary & Topology - CoW only)
        l_c_cl = self.cow_cl_loss(preds['cow'], targets[:, 2:3]) if self.cow_cl_w > 0.0 else torch.tensor(0.0, device=targets.device)

        # Lesion Slice-level Classification Loss (per-slice vector)
        lesion_cls_logit = preds.get('lesion_cls', None)
        if lesion_cls_logit is not None:
            lesion_cls_logit_flat = lesion_cls_logit.view(-1)  # (B,)
            pos_w = self.lesion_cls_loss_fn.pos_weight.to(targets.device)
            l_l_cls = nn.functional.binary_cross_entropy_with_logits(
                lesion_cls_logit_flat, has_lesion,
                pos_weight=pos_w,
                reduction='none'
            )
            combined_lesion_loss = (1.0 - self.lesion_cls_w) * l_l_m + self.lesion_cls_w * l_l_cls
        else:
            combined_lesion_loss = l_l_m

        # Dynamic slice-level weights (using configurable slice_pos_weight, negative slices default to 1.0)
        lesion_slice_weights = has_lesion * (self.lesion_slice_pos_w - 1.0) + 1.0
        lvo_slice_weights = has_lvo * (self.lvo_slice_pos_w - 1.0) + 1.0

        # Apply weights and take mean to get scalar losses
        combined_lesion_loss_scalar = (combined_lesion_loss * lesion_slice_weights).mean()
        l_v_m_scalar = (l_v_m * lvo_slice_weights).mean()

        # 3. Final Task Weighting
        loss_l = combined_lesion_loss_scalar * p_l
        loss_v = l_v_m_scalar * p_v
        loss_c = ((1.0 - self.cow_cl_w) * l_c_m + self.cow_cl_w * l_c_cl) * p_c

        # Unweighted task losses (dùng để logging/monitoring độ hội tụ thực tế)
        unweighted_lesion = combined_lesion_loss_scalar
        unweighted_lvo = l_v_m_scalar
        unweighted_cow = ((1.0 - self.cow_cl_w) * l_c_m + self.cow_cl_w * l_c_cl)
        unweighted_main = unweighted_lesion + unweighted_lvo + unweighted_cow

        main_loss = loss_l + loss_v + loss_c

        aux_l = torch.tensor(0.0, device=targets.device)
        aux_v = torch.tensor(0.0, device=targets.device)
        aux_c = torch.tensor(0.0, device=targets.device)
        
        aux_dict = preds.get("aux_masks", {})
        # Duyệt qua từng task trong dictionary AUX
        for task_key, aux_list in aux_dict.items():
            task_aux_loss = 0.0
            num_active = 0
            for a_p in aux_list:
                if a_p is None: continue # Bỏ qua các tầng bị tắt
                num_active += 1
                
                h, w = a_p.shape[2:]
                
                if task_key == "lesion":
                    t_l = F.interpolate(targets[:, 0:1].float(), (h, w), mode='area')
                    # Dùng sigma_floor cho aux (không cần curriculum, chỉ cần gradient ổn định)
                    aux_focal = self.lesion_main_loss(a_p, t_l, sigma=self.lesion_sigma_floor)
                    aux_dice = self.lesion_dice_loss_fn(a_p, t_l)
                    task_aux_loss += (1.0 - self.lesion_dice_w) * aux_focal + self.lesion_dice_w * aux_dice
                
                elif task_key == "lvo":
                    t_v = F.adaptive_max_pool2d(targets[:, 1:2], (h, w))
                    if isinstance(self.lvo_loss_fn, ModifiedFocalLoss):
                        task_aux_loss += self.lvo_loss_fn(a_p, t_v, sigma=4.0)
                    else:
                        task_aux_loss += self.lvo_loss_fn(a_p, t_v)
                    
                elif task_key == "cow":
                    t_c = F.interpolate(targets[:, 2:3].float(), (h, w), mode='nearest')
                    task_aux_loss += self.cow_main_loss(a_p, t_c)
            
            if num_active > 0:
                # Trọng số Aux = 0.5 * TaskWeight, đã chia trung bình qua các tầng
                if task_key == "lesion":
                    task_aux_loss_weighted = (task_aux_loss * lesion_slice_weights).mean()
                    aux_l = (task_aux_loss_weighted / num_active) * p_l * 0.5
                elif task_key == "lvo":
                    task_aux_loss_weighted = (task_aux_loss * lvo_slice_weights).mean()
                    aux_v = (task_aux_loss_weighted / num_active) * p_v * 0.5
                elif task_key == "cow":
                    aux_c = (task_aux_loss / num_active) * p_c * 0.5

        aux_loss = aux_l + aux_v + aux_c
        total = main_loss + aux_loss
        
        # Tính tổng loss của từng nhiệm vụ
        total_lesion = loss_l + aux_l
        total_lvo = loss_v + aux_v
        total_cow = loss_c + aux_c

        return {
            'total': total, 'main': main_loss, 'aux': aux_loss,
            'total_lesion': total_lesion, 'total_lvo': total_lvo, 'total_cow': total_cow,
            'l_lesion': loss_l.item() if isinstance(loss_l, torch.Tensor) else loss_l, 
            'l_lvo': loss_v.item() if isinstance(loss_v, torch.Tensor) else loss_v, 
            'l_cow': loss_c.item() if isinstance(loss_c, torch.Tensor) else loss_c,
            'p_lesion': p_l if isinstance(p_l, (float, int)) else p_l.item(),
            'p_lvo': p_v if isinstance(p_v, (float, int)) else p_v.item(),
            'p_cow': p_c if isinstance(p_c, (float, int)) else p_c.item(),
            'unweighted_main': unweighted_main.item() if isinstance(unweighted_main, torch.Tensor) else unweighted_main
        }
