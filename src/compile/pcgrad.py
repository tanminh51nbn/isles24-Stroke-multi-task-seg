import torch
import random
import contextlib
from typing import List

class PCGrad:
    def __init__(self, optimizer, use_amp: bool = True):
        self.optimizer = optimizer
        self.use_amp = use_amp

    def backward(self, losses: List[torch.Tensor], model, scaler=None):
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
            for loss in losses:
                self.optimizer.zero_grad(set_to_none=True)
                if scaler is not None and self.use_amp:
                    scaler.scale(loss).backward(retain_graph=True)
                else:
                    loss.backward(retain_graph=True)
                
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
        
        # Flatten gradients của từng task để tính tích vô hướng toàn cục (Global Dot Product)
        task_flat_grads = []
        for grads in task_grads:
            flat_g = torch.cat([g.view(-1) for g in grads])
            task_flat_grads.append(flat_g)

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

        # Nếu chạy DDP, thực hiện đồng bộ thủ công (all-reduce) cho gradient cuối cùng
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(final_flat_grad, op=torch.distributed.ReduceOp.SUM)
            final_flat_grad /= torch.distributed.get_world_size()

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
