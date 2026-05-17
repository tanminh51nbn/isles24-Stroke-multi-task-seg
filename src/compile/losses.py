"""
losses.py — Task-specific Loss Functions cho Multi-Task Stroke Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple

# ─── Tversky Loss ─────────────────────────────────────────────────────────────

class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
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
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 2.0, smooth: float = 1.0):
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
        error = (1.0 - tversky_index).clamp(min=1e-6, max=1.0)
        return torch.pow(error, self.gamma).mean()

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
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sigma: float = 0.0, debug: bool = False) -> torch.Tensor:
        pred = torch.sigmoid(logits.float()).clamp(min=self.eps, max=1.0 - self.eps)
        
        # 1. Vùng Đỏ (Nhãn gốc) và Vùng Nền (Tất cả còn lại)
        # Sử dụng nguyên gốc targets thay vì đỉnh của heatmap
        pos_mask = (targets >= 0.5).float()
        neg_mask = 1.0 - pos_mask
        
        # 2. Bản đồ Không gian (Heatmap lan tỏa)
        heatmap_gt = apply_gaussian_blur(targets, sigma) if sigma > 0 else targets
        
        # 3. Tính Lực Thưởng và Lực Phạt
        # Lực phạt (neg_loss) bị triệt tiêu bằng (1 - heatmap_gt)^beta khi tiến gần Vùng Đỏ
        # [FIX NaN] Thêm self.eps để tránh log(0) gây ra NaN/Inf
        pos_loss = -pos_mask * torch.pow(1.0 - pred, self.alpha) * torch.log(pred + self.eps)
        neg_loss = -neg_mask * torch.pow(1.0 - heatmap_gt, self.beta) * torch.pow(pred, self.alpha) * torch.log(1.0 - pred + self.eps)
        
        # 4. Đòn bẩy chống khôn lỏi (Dynamic Force Balancing)
        num_pos = pos_mask.sum().clamp(min=1.0)
        num_neg = neg_mask.sum().clamp(min=1.0)
        
        # [FIX] Trừng phạt thiết quân luật: Giảm giới hạn max từ 1000 xuống 50
        # Mục đích: Không cho mô hình "vẽ bừa" để đổi lấy điểm TP nữa.
        weight_pos = (num_neg / num_pos).clamp(max=50.0)
        
        # Bơm sức mạnh cho Vùng Đỏ bằng đòn bẩy
        loss_pos = pos_loss.sum() * weight_pos
        loss_neg = neg_loss.sum()
        
        # Chia trung bình có trọng số để loss scale ổn định [0, 1]
        loss = (loss_pos + loss_neg) / (num_neg + num_pos * weight_pos)
        
        if debug:
            num_pos_val = int(pos_mask.sum().item())
            weight_val = weight_pos.item()
            print(f"      [MFL_DEBUG] pos_px={num_pos_val} weight_pos={weight_val:.1f} "
                  f"pos_loss={loss_pos.item():.2f} neg_loss={loss_neg.item():.2f} total={loss.item():.4f}")

        return torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=0.0).clamp(max=100.0)

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
    def forward(self, logits: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        fp_loss = probs * torch.clamp(sdf, min=0)
        fn_loss = (1 - probs) * torch.abs(torch.clamp(sdf, max=0))
        return (fp_loss + fn_loss).mean()

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
    def __init__(self, iters: int = 3, smooth: float = 1e-5):
        super().__init__()
        self.iters, self.smooth = iters, smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        t_s, p_s = soft_skel(targets, self.iters), soft_skel(probs, self.iters)
        t_p = ((p_s * targets).sum(dim=(1, 2, 3)) + self.smooth) / (p_s.sum(dim=(1, 2, 3)) + self.smooth)
        t_s_ = ((t_s * probs).sum(dim=(1, 2, 3)) + self.smooth) / (t_s.sum(dim=(1, 2, 3)) + self.smooth)
        return 1.0 - ((2.0 * t_p * t_s_) / (t_p + t_s_ + self.smooth)).mean()

# ─── Multi-Task Loss (The Core) ───────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        l_cfg  = config["loss"]
        t_cfg  = config.get("training", {})
        self.init_w = t_cfg.get("initial_weights", {"lesion": 1.0, "lvo": 1.0, "cow": 1.0})

        # 1. Lesion Task
        self.lesion_main_loss = FocalTverskyLoss(
            alpha=l_cfg["lesion"].get("alpha", 0.7),
            beta=l_cfg["lesion"].get("beta", 0.3),
            gamma=l_cfg["lesion"].get("gamma", 2.0)
        )
        self.lesion_hd_loss = SDFBoundaryLoss()
        self.lesion_hd_w = l_cfg["lesion"].get("hd_weight", 0.0)

        # 2. LVO Task
        l_v_cfg = l_cfg.get("lvo", {})
        lvo_type = l_v_cfg.get("type", "modified_focal")
        if lvo_type == "focal_tversky":
            self.lvo_loss_fn = FocalTverskyLoss(
                alpha=l_v_cfg.get("alpha", 0.7),
                beta=l_v_cfg.get("beta", 0.3),
                gamma=l_v_cfg.get("gamma", 2.0)
            )
        else:
            self.lvo_loss_fn = ModifiedFocalLoss(
                alpha=l_v_cfg.get("mfl_alpha", 2.0),
                beta=l_v_cfg.get("mfl_beta", 4.0)
            )
        # [T2.1] LVO Binary Classification Loss
        # [FIX] Hạ pos_weight từ 5.0 xuống 1.5 để dập tắt ảo giác (giảm >1200 FPs)
        lvo_cls_pos_w = l_v_cfg.get("lvo_cls_pos_weight", 1.5)
        self.lvo_cls_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([lvo_cls_pos_w])
        )
        self.lvo_cls_w = l_v_cfg.get("lvo_cls_weight", 0.3)  # Tỷ lệ cls trong tổng LVO loss

        # 3. CoW Task
        self.cow_main_loss = FocalTverskyLoss(
            alpha=l_cfg["cow"].get("alpha", 0.5),
            beta=l_cfg["cow"].get("beta", 0.5),
            gamma=l_cfg["cow"].get("gamma", 2.0)
        )
        self.cow_cl_loss = SoftCLDiceLoss(iters=l_cfg["cow"].get("iters", 3))
        self.cow_cl_w = l_cfg["cow"].get("cl_weight", 0.0)

        # ── Performance Gap Weighting (PGW) ────────────────────────────────────
        # Thay thế DWA+: điều chỉnh trọng số dựa trên khoảng cách đến target metric,
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
        self.pgw_w_min        = pgw_cfg.get("w_min",       0.1)
        self.pgw_start_epoch  = pgw_cfg.get("start_epoch", 5)
        self.n_tasks          = 3

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

        Gọi 1 lần/epoch từ trainer.py (rank 0) SAU KHI validate() xong.
        current_weights được cập nhật in-place → forward() chỉ đọc, không tính lại.

        Nguyên lý:
            gap_i = max(0, target_i - current_i)   # Khoảng cách còn lại đến mục tiêu
            w_i   = softmax(gap / T) * N            # Task kém nhất → weight cao nhất
            Sau clamp(w_min) → renormalize sum=N    # Giữ gradient scale ổn định
        """
        if epoch < self.pgw_start_epoch:
            return  # Giữ initial_weights trong giai đoạn warmup

        current = torch.tensor([
            val_metrics.get("dice_lesion", 0.0),
            val_metrics.get("f1_lvo",    0.0) / 100.0,  # F1 → [0,1]
            val_metrics.get("dice_cow",  0.0),
        ], device=self.current_weights.device)

        target = self.perf_targets.to(current.device)
        gap = torch.clamp(target - current, min=0.0)  # Chỉ tính gap âm (chưa đạt)

        if gap.sum() < 1e-6:
            # Tất cả task đã vượt target → dùng lại initial_weights (không có task nào cần ưu tiên đặc biệt)
            init = torch.tensor(
                [self.init_w['lesion'], self.init_w['lvo'], self.init_w['cow']],
                device=self.current_weights.device
            )
            self.current_weights.copy_(init)
            print(f"    [PGW] Tất cả task đạt target → reset về initial_weights")
            return

        # [FIX 1.2] Tách task ĐÃ đạt target (gap≈0) và CHƯA đạt.
        # Task đã đạt → giữ cố định ở w_min, KHÔNG tham gia softmax.
        # Ngăn CoW nhận weight 0.57 khi gap_C≈0.
        active_mask = gap > 1e-3
        n_active   = int(active_mask.sum().item())
        n_inactive = self.n_tasks - n_active

        if n_active == 0:
            init = torch.tensor(
                [self.init_w['lesion'], self.init_w['lvo'], self.init_w['cow']],
                device=self.current_weights.device
            )
            self.current_weights.copy_(init)
            return

        # Softmax chỉ trên active tasks; budget = N - n_inactive * w_min
        weight_budget = self.n_tasks - n_inactive * self.pgw_w_min
        active_gap    = gap * active_mask.float()
        # [FIX NaN] Subtract max before exp (Softmax stability trick) để tránh nổ Inf khi gap lớn hoặc temp nhỏ
        max_gap       = torch.max(active_gap) if n_active > 0 else 0.0
        exp_gap       = torch.exp((active_gap - max_gap) / self.pgw_temperature) * active_mask.float()
        raw_w = torch.where(
            active_mask,
            (exp_gap / (exp_gap.sum() + 1e-8)) * weight_budget,
            torch.full_like(gap, self.pgw_w_min)
        )

        # Momentum + clamp + renormalize
        blended_w = self.pgw_momentum * self.current_weights + (1.0 - self.pgw_momentum) * raw_w
        clamped_w = torch.clamp(blended_w, min=self.pgw_w_min)
        final_w   = clamped_w * (self.n_tasks / clamped_w.sum())
        self.current_weights.copy_(final_w)
        self._pgw_epoch.fill_(epoch)

        fixed_tag = "".join([
            "L" if not active_mask[0] else "",
            "V" if not active_mask[1] else "",
            "C" if not active_mask[2] else "",
        ]) or "none"
        print(f"    [PGW]: Gap L={gap[0]:.3f} V={gap[1]:.3f} C={gap[2]:.3f} "
              f"→ Weights: L={final_w[0]:.2f} V={final_w[1]:.2f} C={final_w[2]:.2f} "
              f"(fixed: {fixed_tag})")

    def forward(self, preds: dict, targets: torch.Tensor, epoch: int, **kwargs) -> dict:
        cur_ep = int(epoch)
        # Chỉ ĐỌC current_weights — được PGW cập nhật 1 lần/epoch từ update_weights_from_metrics().
        # Trước pgw_start_epoch: current_weights = initial_weights (set trong __init__).
        w = self.current_weights.to(targets.device)
        p_l, p_v, p_c = w[0], w[1], w[2]

        dynamic_sigma = max(1.5, 5.0 * (0.92 ** cur_ep))
        if kwargs.get('batch_idx', 0) == 0 and (not dist.is_initialized() or dist.get_rank()==0):
            pgw_active = self._pgw_epoch.item() >= 0
            tag = "[PGW]" if pgw_active else "[INIT]"
            print(f"    {tag}: Weights [Lesion={p_l:.2f}, LVO={p_v:.2f}, CoW={p_c:.2f}] | Sigma={dynamic_sigma:.2f}")

        # 1. Main Losses
        _debug_lvo = (kwargs.get('batch_idx', -1) == 0) and (not dist.is_initialized() or dist.get_rank() == 0)
        l_l_m = self.lesion_main_loss(preds['lesion'], targets[:, 0:1])
        
        if isinstance(self.lvo_loss_fn, ModifiedFocalLoss):
            l_v_m = self.lvo_loss_fn(preds['lvo'], targets[:, 1:2], sigma=dynamic_sigma, debug=_debug_lvo)
        else:
            # Focal Tversky Loss on Heatmap
            heatmap_gt = apply_gaussian_blur(targets[:, 1:2], dynamic_sigma) if dynamic_sigma > 0 else targets[:, 1:2]
            l_v_m = self.lvo_loss_fn(preds['lvo'], heatmap_gt)
            if _debug_lvo:
                num_pos = (targets[:, 1:2] > 0.5).sum().item()
                print(f"      [LVO_DEBUG] PosPixels: {int(num_pos)} | FTL_Loss: {l_v_m.item():.4f}")
        l_c_m = self.cow_main_loss(preds['cow'], targets[:, 2:3])

        # [T2.1] LVO Binary Classification Loss
        # has_lvo = 1 nếu slice này có bất kỳ pixel LVO GT nào (global max > 0)
        has_lvo = (targets[:, 1:2].amax(dim=(1, 2, 3), keepdim=False) > 0).float()  # (B,)
        lvo_cls_logit = preds.get('lvo_cls', None)
        if lvo_cls_logit is not None:
            lvo_cls_logit_flat = lvo_cls_logit.view(-1)  # (B,)
            pos_w = self.lvo_cls_loss_fn.pos_weight.to(targets.device)
            l_v_cls = nn.functional.binary_cross_entropy_with_logits(
                lvo_cls_logit_flat, has_lvo,
                pos_weight=pos_w
            )
            l_v_m = (1.0 - self.lvo_cls_w) * l_v_m + self.lvo_cls_w * l_v_cls

        # 2. Additional Task-specific Losses (Boundary & Topology)
        l_l_hd = self.lesion_hd_loss(preds['lesion'], targets[:, 3:4])
        l_c_cl = self.cow_cl_loss(preds['cow'], targets[:, 2:3])

        # [T2.3] Asymmetric FP penalty nhỏ cho Lesion (phạt nặng pixel FP confident)
        probs_l = torch.sigmoid(preds['lesion'].float())
        afl_fp = -(1 - targets[:, 0:1]) * torch.pow(probs_l, 2.0) * torch.log(1 - probs_l + 1e-6)
        l_l_afl = 0.1 * afl_fp.mean()

        # 3. Final Task Weighting
        loss_l = ((1.0 - self.lesion_hd_w) * l_l_m + self.lesion_hd_w * l_l_hd) * p_l + l_l_afl
        loss_v = l_v_m * p_v
        loss_c = ((1.0 - self.cow_cl_w) * l_c_m + self.cow_cl_w * l_c_cl) * p_c

        main_loss = loss_l + loss_v + loss_c

        # (running_loss không còn dùng cho DWA — giữ lại để debug nếu cần)

        aux_loss = 0.0
        # 2. Auxiliary Losses (Multi-Level)
        aux_dict = preds.get("aux_masks", {})
        
        # Duyệt qua từng task trong dictionary AUX
        for task_key, aux_list in aux_dict.items():
            for a_p in aux_list:
                if a_p is None: continue # Bỏ qua các tầng bị tắt (như LVO 16x16, 32x32)
                
                h, w = a_p.shape[2:]
                
                if task_key == "lesion":
                    t_l = F.interpolate(targets[:, 0:1].float(), (h, w), mode='nearest')
                    aux_loss += self.lesion_main_loss(a_p, t_l) * p_l * 0.5
                
                elif task_key == "lvo":
                    t_v = F.adaptive_max_pool2d(targets[:, 1:2], (h, w))
                    if isinstance(self.lvo_loss_fn, ModifiedFocalLoss):
                        aux_loss += self.lvo_loss_fn(a_p, t_v, sigma=4.0) * p_v * 0.5
                    else:
                        aux_loss += self.lvo_loss_fn(a_p, t_v) * p_v * 0.5
                    
                elif task_key == "cow":
                    t_c = F.interpolate(targets[:, 2:3].float(), (h, w), mode='nearest')
                    aux_loss += self.cow_main_loss(a_p, t_c) * p_c * 0.5

        total = main_loss + aux_loss
        return {
            'total': total, 'main': main_loss, 'aux': aux_loss,
            'l_lesion': loss_l.item() if isinstance(loss_l, torch.Tensor) else loss_l, 
            'l_lvo': loss_v.item() if isinstance(loss_v, torch.Tensor) else loss_v, 
            'l_cow': loss_c.item() if isinstance(loss_c, torch.Tensor) else loss_c,
            'p_lesion': p_l if isinstance(p_l, (float, int)) else p_l.item(),
            'p_lvo': p_v if isinstance(p_v, (float, int)) else p_v.item(),
            'p_cow': p_c if isinstance(p_c, (float, int)) else p_c.item()
        }
