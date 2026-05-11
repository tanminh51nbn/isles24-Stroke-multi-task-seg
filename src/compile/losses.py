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
        
        # 1. Tính toán loss thô từng pixel
        pos_loss = -pos_mask * torch.pow(1.0 - pred, self.alpha) * torch.log(pred)
        neg_loss = -neg_mask * torch.pow(1.0 - heatmap_gt, self.beta) * torch.pow(pred, self.alpha) * torch.log(1.0 - pred)
        
        # 2. Positive & Negative Loss — công thức gốc CenterNet (Law & Deng, 2019)
        # Tổng cả hai nhánh chia chung cho num_pos để scale đồng nhất.
        # Modified Focal Loss đã là "soft miner" tự nhiên qua (1-heatmap)^beta * pred^alpha,
        # không cần Top-K — Top-K chỉ khiến 99%+ pixel âm tính mất gradient (FP không bị phạt).
        num_pos   = pos_mask.sum().clamp(min=1.0)
        loss_pos  = pos_loss.sum()
        loss_neg  = neg_loss.sum()
        loss = (loss_pos + loss_neg) / num_pos
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
        # [FIX Bug2] Đọc đúng từ key 'training', không phải 'curriculum_learning' (key không tồn tại)
        self.dwa_start = t_cfg.get("dwa_start_epoch", 25)
        self.init_w    = t_cfg.get("initial_weights", {"lesion": 1.0, "lvo": 1.0, "cow": 1.0})
        
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
        
        # [STABLE DWA+ Parameters]
        self.register_buffer('ema_loss', torch.ones(3))
        self.register_buffer('prev_ema_loss', torch.ones(3))
        self.register_buffer('current_weights', torch.ones(3))
        
        # Load hyperparams từ config (t_cfg đã khai báo ở trên, tránh khai báo lại)
        self.temp  = t_cfg.get("dwa_temperature", 2.0)
        self.alpha = t_cfg.get("dwa_ema_alpha", 0.2)   # EMA smoothing factor
        self.beta  = t_cfg.get("dwa_momentum", 0.5)    # Weight momentum factor
        
        # Gán trọng số ban đầu vào current_weights
        with torch.no_grad():
            self.current_weights[0] = self.init_w['lesion']
            self.current_weights[1] = self.init_w['lvo']
            self.current_weights[2] = self.init_w['cow']
        
        self.running_loss = [0.0, 0.0, 0.0]
        self.running_counts = 0

    def update_epoch_stats(self, epoch: int):
        """Cập nhật EMA Loss sau mỗi Epoch, sau đó refresh DWA weights 1 lần.
        
        [FIX Bug3] Nhận epoch làm tham số để _refresh_dwa_weights biết khi nào kích hoạt.
        Tách hoàn toàn việc tính trọng số khỏi forward() — đảm bảo nhất quán DDP.
        """
        if self.running_counts > 0:
            # 1. Tính loss trung bình thực tế của epoch vừa qua
            curr_avg = torch.tensor([l/self.running_counts for l in self.running_loss], device=self.ema_loss.device)
            if dist.is_initialized():
                dist.all_reduce(curr_avg, op=dist.ReduceOp.SUM)
                curr_avg = curr_avg / dist.get_world_size()
            
            # 2. Cập nhật EMA Loss: L_ema = alpha * L_curr + (1 - alpha) * L_ema_old
            self.prev_ema_loss.copy_(self.ema_loss)
            self.ema_loss.copy_(self.alpha * curr_avg + (1.0 - self.alpha) * self.ema_loss)
            
            # Reset bộ đếm batch
            self.running_loss, self.running_counts = [0.0, 0.0, 0.0], 0

        # 3. Refresh DWA weights 1 lần/epoch (sau khi EMA đã được cập nhật)
        self._refresh_dwa_weights(epoch)

    def _refresh_dwa_weights(self, epoch: int):
        """Cập nhật current_weights theo DWA — chỉ gọi 1 lần/epoch từ update_epoch_stats().

        [FIX Bug3] Tách hoàn toàn khỏi forward() để:
        - Tránh ghi đè state n_batches lần/epoch (sai momentum).
        - Đảm bảo 2 GPU DDP dùng cùng trọng số (current_weights là buffer, không all_reduce
          trong forward, nhưng nhất quán vì chỉ update 1 lần sau all_reduce EMA).
        """
        if epoch < self.dwa_start + 1:
            return  # Giữ nguyên init_w cho đến khi đủ epoch

        # 1. Tỉ số tiến bộ: r > 1 = task đang giảm chậm hơn epoch trước
        r = self.ema_loss / (self.prev_ema_loss + 1e-8)

        # 2. Softmax với temperature → phân phối trọng số (tổng = 3.0)
        exp_r = torch.exp((r - torch.max(r)) / self.temp)
        new_weights = (exp_r / (exp_r.sum() + 1e-8)) * 3.0

        # 3. Momentum: dịch chuyển mượt mà, chống nhảy vọt
        updated_weights = self.beta * self.current_weights + (1.0 - self.beta) * new_weights
        self.current_weights.copy_(updated_weights)

    def forward(self, preds: dict, targets: torch.Tensor, epoch: int, **kwargs) -> dict:
        cur_ep = int(epoch)
        # [FIX Bug3] Chỉ ĐỌC current_weights — không tính lại DWA mỗi batch.
        # Trước dwa_start: current_weights = init_w (set trong __init__).
        # Sau dwa_start:  current_weights được _refresh_dwa_weights() cập nhật 1 lần/epoch.
        w = self.current_weights.to(targets.device)
        p_l, p_v, p_c = w[0], w[1], w[2]

        if kwargs.get('batch_idx', 0) == 0 and (not dist.is_initialized() or dist.get_rank()==0):
            print(f"    [LOSS_POLICY] Ep {cur_ep}: CoW={p_c:.2f}, LVO={p_v:.2f}, Lesion={p_l:.2f}")

        # 1. Main Losses
        l_l_m = self.lesion_main_loss(preds['lesion'], targets[:, 0:1])
        l_v_m = self.lvo_loss_fn(preds['lvo'], targets[:, 1:2], sigma=4.0)
        l_c_m = self.cow_main_loss(preds['cow'], targets[:, 2:3])

        # 2. Additional Task-specific Losses (Boundary & Topology)
        l_l_hd = self.lesion_hd_loss(preds['lesion'], targets[:, 3:4])
        l_c_cl = self.cow_cl_loss(preds['cow'], targets[:, 2:3])

        # 3. Final Task Weighting (Weighted Average + DWA+)
        # [FORMULA UPDATE]: Sử dụng Weighted Average (1-w)*Main + w*Aux để cân bằng Scale
        loss_l = ((1.0 - self.lesion_hd_w) * l_l_m + self.lesion_hd_w * l_l_hd) * p_l
        loss_v = l_v_m * p_v
        loss_c = ((1.0 - self.cow_cl_w) * l_c_m + self.cow_cl_w * l_c_cl) * p_c
        
        main_loss = loss_l + loss_v + loss_c

        with torch.no_grad():
            self.running_loss[0]+=l_l_m.item(); self.running_loss[1]+=l_v_m.item(); self.running_loss[2]+=l_c_m.item()
            self.running_counts+=1

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
                    aux_loss += self.lvo_loss_fn(a_p, t_v, sigma=4.0) * p_v * 0.5
                    
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
