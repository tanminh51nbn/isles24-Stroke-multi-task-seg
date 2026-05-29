import torch
import random
import contextlib
from typing import List

class PCGrad:
    def __init__(self, optimizer, use_amp: bool = True, max_norm: float = 10.0):
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.max_norm = max_norm
        self._enc_debug = None  # [DEBUG] Encoder gradient analysis

    def backward(self, losses: List[torch.Tensor], model, scaler=None, encoder_debug_ids=None):
        """
        Thực hiện lan truyền ngược có phẫu thuật gradient (Gradient Surgery)
        để giải quyết xung đột hướng gradient giữa các task.
        """
        # 1. Thu thập danh sách các tham số cần tối ưu và có requires_grad
        params = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    params.append(p)

        # 2. Lan truyền ngược riêng lẻ từng task và lưu lại gradient
        task_grads = []
        
        # Nếu chạy DDP, tắt tự động đồng bộ gradient lúc backward từng task
        is_ddp = hasattr(model, "no_sync") and torch.distributed.is_initialized()
        context = model.no_sync() if is_ddp else contextlib.nullcontext()
        
        with context:
            for idx, loss in enumerate(losses):
                self.optimizer.zero_grad(set_to_none=True)
                retain = (idx < len(losses) - 1)
                if scaler is not None and self.use_amp:
                    scaler.scale(loss).backward(retain_graph=retain)
                else:
                    loss.backward(retain_graph=retain)
                
                # Lưu gradient dưới dạng list của từng tensor
                grads = []
                for p in params:
                    if p.grad is not None:
                        grads.append(p.grad.clone())
                    else:
                        # Nếu tham số không nhận gradient từ task này, gán bằng 0
                        grads.append(torch.zeros_like(p))
                task_grads.append(grads)

        # Xóa gradient tạm thời
        self.optimizer.zero_grad(set_to_none=True)

        if not task_grads:
            return

        # 3. Phẫu thuật Gradient (PCGrad Projection)
        num_tasks = len(losses)
        
        # Flatten gradients của từng task và đồng bộ hóa qua DDP trước khi phẫu thuật
        task_flat_grads = []
        for grads in task_grads:
            flat_g = torch.cat([g.view(-1) for g in grads])
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(flat_g, op=torch.distributed.ReduceOp.SUM)
                flat_g /= torch.distributed.get_world_size()
            task_flat_grads.append(flat_g)

        # [DEBUG] Phân tích gradient per-task trên Encoder (trước PCGrad projection)
        if encoder_debug_ids is not None:
            self._analyze_encoder_grads(params, task_flat_grads, encoder_debug_ids, scaler)

        # [MAGNITUDE BALANCING] Cắt gọt độ lớn (Gradient Clipping) trên từng task riêng biệt
        # Ngăn chặn một task (như LVO) dùng độ lớn khổng lồ để lấn át các task khác trước khi xét hướng.
        scale = scaler.get_scale() if scaler is not None else 1.0

        if self.max_norm is not None and self.max_norm > 0:
            for i in range(num_tasks):
                scaled_norm = task_flat_grads[i].norm(2)
                unscaled_norm = scaled_norm / scale
                
                # Nếu gradient bị overflow (inf/nan), bỏ qua clip để Scaler của PyTorch tự bắt lỗi và skip batch
                if not torch.isfinite(unscaled_norm):
                    continue
                    
                if unscaled_norm > self.max_norm:
                    task_flat_grads[i] = task_flat_grads[i] * (self.max_norm / (unscaled_norm + 1e-8))

        projected_flat_grads = []
        for i in range(num_tasks):
            g_i = task_flat_grads[i].clone()
            # Trộn ngẫu nhiên thứ tự các task khác để tránh thiên vị (bias)
            other_indices = list(range(num_tasks))
            other_indices.remove(i)
            random.shuffle(other_indices)

            for j in other_indices:
                g_j = task_flat_grads[j]
                dot_prod = torch.dot(g_i, g_j)
                if dot_prod < 0:
                    # Phát hiện xung đột! Chiếu g_i lên mặt phẳng trực giao với g_j
                    g_j_norm_sq = torch.dot(g_j, g_j) + 1e-8
                    g_i -= (dot_prod / g_j_norm_sq) * g_j
            projected_flat_grads.append(g_i)

        # 4. Cộng dồn các gradient đã qua phẫu thuật
        final_flat_grad = torch.stack(projected_flat_grads).sum(dim=0)

        # 5. Khôi phục (Unflatten) gradient về lại thuộc tính p.grad của từng tham số
        offset = 0
        for p in params:
            numel = p.numel()
            grad_slice = final_flat_grad[offset:offset+numel].view_as(p)
            if p.grad is None:
                p.grad = grad_slice.clone()
            else:
                p.grad.copy_(grad_slice)
            offset += numel

    def _analyze_encoder_grads(self, params, task_flat_grads, enc_ids, scaler=None):
        """Phân tích gradient per-task trên Encoder params (trước PCGrad projection)."""
        enc_ranges = []
        offset = 0
        for p in params:
            numel = p.numel()
            if id(p) in enc_ids:
                enc_ranges.append((offset, offset + numel))
            offset += numel

        if not enc_ranges:
            self._enc_debug = None
            return

        scale = scaler.get_scale() if scaler is not None else 1.0
        
        task_enc_grads = []
        for flat_g in task_flat_grads:
            parts = [flat_g[start:end] for start, end in enc_ranges]
            task_enc_grads.append(torch.cat(parts))

        names = ["Lesion", "LVO", "CoW"]
        norms = {n: (g.norm(2).item() / scale) for n, g in zip(names, task_enc_grads)}

        cosine = {}
        for i, j, key in [(0, 1, "L,V"), (0, 2, "L,C"), (1, 2, "V,C")]:
            cos = torch.nn.functional.cosine_similarity(
                task_enc_grads[i].unsqueeze(0),
                task_enc_grads[j].unsqueeze(0)
            ).item()
            cosine[key] = cos

        self._enc_debug = {"norms": norms, "cosine": cosine}
