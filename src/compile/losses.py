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
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 2.0, smooth: float = 1e-5):
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

# ─── Modified Focal Loss ──────────────────────────────────────────────────────

class ModifiedFocalLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    def _apply_gaussian_blur(self, mask: torch.Tensor, sigma: float) -> torch.Tensor:
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
        return blurred / peaks.clamp(min=1e-6)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sigma: float = 0.0) -> torch.Tensor:
        pred = torch.sigmoid(logits.float()).clamp(min=self.eps, max=1.0 - self.eps)
        heatmap_gt = self._apply_gaussian_blur(targets, sigma) if sigma > 0 else targets
        pos_mask = (heatmap_gt == 1.0).float()
        neg_mask = 1.0 - pos_mask
        pos_loss = -pos_mask * torch.pow(1.0 - pred, self.alpha) * torch.log(pred)
        neg_loss = -neg_mask * torch.pow(1.0 - heatmap_gt, self.beta) * torch.pow(pred, self.alpha) * torch.log(1.0 - pred)
        num_pos = pos_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)

        # [FIX] Tách biệt cách tính để tránh khuếch đại nhiễu nền (Background Noise)
        # loss_pos: Chia cho số lượng điểm dương để đảm bảo tín hiệu điểm tắc đủ mạnh
        loss_pos = pos_loss.sum(dim=(1, 2, 3)) / num_pos
        # loss_neg: Dùng mean để tránh việc sum của 65k pixels bị chia cho 1 (gây nổ Gradient)
        loss_neg = neg_loss.mean(dim=(1, 2, 3))
        
        loss = (loss_pos + loss_neg).mean()
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
        l_cfg = config["loss"]
        self.curr_cfg = config["training"].get("curriculum_learning", {})
        self.enabled = self.curr_cfg.get("enabled", False)
        self.dwa_start = self.curr_cfg.get("dwa_start_epoch", 25)
        self.init_w = self.curr_cfg.get("initial_weights", {"lesion": 1.0, "lvo": 1.0, "cow": 1.0})
        
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
        self.lvo_loss_fn = ModifiedFocalLoss(
            alpha=l_v_cfg.get("mfl_alpha", 2.0), 
            beta=l_v_cfg.get("mfl_beta", 4.0)
        )

        # 3. CoW Task
        self.cow_main_loss = FocalTverskyLoss(
            alpha=l_cfg["cow"].get("alpha", 0.5),
            beta=l_cfg["cow"].get("beta", 0.5),
            gamma=l_cfg["cow"].get("gamma", 2.0)
        )
        self.cow_cl_loss = SoftCLDiceLoss(iters=l_cfg["cow"].get("iters", 3))
        self.cow_cl_w = l_cfg["cow"].get("cl_weight", 0.0)
        
        self.log_vars = nn.ParameterDict({
            'lesion': nn.Parameter(torch.tensor(0.5)),
            'lvo': nn.Parameter(torch.tensor(0.5)),
            'cow': nn.Parameter(torch.tensor(0.5))
        })
        
        self.register_buffer('prev_loss', torch.ones(3))
        self.register_buffer('prev_loss_2', torch.ones(3))
        
        u_cfg = config.get("uncertainty", {})
        self.temp = u_cfg.get("dwa_temperature", 2.0)
        self.s_min_lvo = u_cfg.get("s_min_lvo", 0.1)
        self.s_min_lesion = u_cfg.get("s_min_lesion", 0.3)
        self.s_min_cow = u_cfg.get("s_min_cow", 0.6)
        
        self.running_loss = [0.0, 0.0, 0.0]
        self.running_counts = 0

    def update_epoch_stats(self):
        if self.running_counts > 0:
            avg_loss = torch.tensor([l/self.running_counts for l in self.running_loss], device=self.prev_loss.device)
            if dist.is_initialized():
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss / dist.get_world_size()
            self.prev_loss_2.copy_(self.prev_loss)
            self.prev_loss.copy_(avg_loss)
            self.running_loss, self.running_counts = [0.0, 0.0, 0.0], 0

    def _calculate_dwa(self, epoch):
        if epoch < 2: return torch.ones(3, device=self.prev_loss.device)
        r = self.prev_loss / (self.prev_loss_2 + 1e-8)
        exp_r = torch.exp((r - torch.max(r)) / self.temp)
        return (exp_r / (exp_r.sum() + 1e-8)) * 3.0

    def forward(self, preds: dict, targets: torch.Tensor, epoch: int, **kwargs) -> dict:
        cur_ep = int(epoch)
        # CHIẾN THUẬT TRỌNG SỐ (DWA+ Mechanism)
        if cur_ep >= self.dwa_start:
            w = self._calculate_dwa(cur_ep).to(targets.device)
            p_l, p_v, p_c = w[0], w[1], w[2]
        else:
            p_l, p_v, p_c = self.init_w['lesion'], self.init_w['lvo'], self.init_w['cow']

        if kwargs.get('batch_idx', 0) == 0 and (not dist.is_initialized() or dist.get_rank()==0):
            print(f"    [LOSS_POLICY] Ep {cur_ep}: CoW={p_c:.2f}, LVO={p_v:.2f}, Lesion={p_l:.2f}")

        # 1. Main Losses
        l_l_m = self.lesion_main_loss(preds['lesion'], targets[:, 0:1])
        l_v_m = self.lvo_loss_fn(preds['lvo'], targets[:, 1:2], sigma=4.0)
        l_c_m = self.cow_main_loss(preds['cow'], targets[:, 2:3])

        # 2. Additional Task-specific Losses (Boundary & Topology)
        # Lesion SDF (từ kênh 3 của target)
        l_l_hd = self.lesion_hd_loss(preds['lesion'], targets[:, 3:4]) * self.lesion_hd_w
        # CoW clDice
        l_c_cl = self.cow_cl_loss(preds['cow'], targets[:, 2:3]) * self.cow_cl_w

        # 3. Uncertainty Weighting
        s_l = torch.clamp(self.log_vars['lesion'], min=self.s_min_lesion, max=10.0)
        s_v = torch.clamp(self.log_vars['lvo'], min=self.s_min_lvo, max=10.0)
        s_c = torch.clamp(self.log_vars['cow'], min=self.s_min_cow, max=10.0)

        loss_l = (torch.exp(-s_l) * (l_l_m + l_l_hd) + s_l) * p_l
        loss_v = (torch.exp(-s_v) * l_v_m + s_v) * p_v
        loss_c = (torch.exp(-s_c) * (l_c_m + l_c_cl) + s_c) * p_c
        main_loss = loss_l + loss_v + loss_c

        with torch.no_grad():
            self.running_loss[0]+=l_l_m.item(); self.running_loss[1]+=l_v_m.item(); self.running_loss[2]+=l_c_m.item()
            self.running_counts+=1

        aux_loss = 0.0
        if "aux_masks" in preds and preds["aux_masks"] is not None:
            a_w = [0.05, 0.075, 0.125, 0.25]
            for i, a_p in enumerate(preds["aux_masks"]):
                if a_p is None or i >= len(a_w): continue
                h, w = a_p.shape[2], a_p.shape[3]
                t_l = F.interpolate(targets[:, 0:1].float(), (h, w), mode='nearest')
                la_l = self.lesion_main_loss(a_p[:, 0:1], t_l) * p_l
                t_v = F.adaptive_max_pool2d(targets[:, 1:2], (h, w))
                la_v = self.lvo_loss_fn(a_p[:, 1:2], t_v, sigma=4.0) * p_v
                t_c = F.interpolate(targets[:, 2:3].float(), (h, w), mode='nearest')
                la_c = self.cow_main_loss(a_p[:, 2:3], t_c) * p_c
                aux_loss += a_w[i] * (la_l + la_v + la_c)

        total = main_loss + aux_loss
        return {
            'total': total, 'main': main_loss, 'aux': aux_loss,
            'l_lesion': loss_l.item(), 'l_lvo': loss_v.item(), 'l_cow': loss_c.item(),
            'p_lesion': p_l if isinstance(p_l, (float, int)) else p_l.item(),
            'p_lvo': p_v if isinstance(p_v, (float, int)) else p_v.item(),
            'p_cow': p_c if isinstance(p_c, (float, int)) else p_c.item(),
            'sigma_lesion': torch.exp(s_l * 0.5).item(),
            'sigma_lvo': torch.exp(s_v * 0.5).item(),
            'sigma_cow': torch.exp(s_c * 0.5).item()
        }
