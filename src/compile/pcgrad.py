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

    def _project_grads(self, task_grads: List[List[torch.Tensor]], params: List[torch.Tensor], scaler=None,
                       weights: List[float] = None, asymmetric: bool = True) -> List[torch.Tensor]:
        """Thực hiện chiếu PCGrad và trả về danh sách gradient đã phẫu thuật."""
        num_tasks = len(task_grads)
        if num_tasks == 0:
            return []
            
        # 1. Flatten gradients của từng task và đồng bộ hóa qua DDP trước khi phẫu thuật
        task_flat_grads = []
        for grads in task_grads:
            flat_g = torch.cat([g.view(-1) for g in grads])
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(flat_g, op=torch.distributed.ReduceOp.SUM)
                flat_g /= torch.distributed.get_world_size()
            task_flat_grads.append(flat_g)

        # Telemetry: Đo conflict và cosine similarity trước khi clipping/projection
        task_names = ["Lesion", "LVO", "CoW"]
        raw_norms = [g.norm(2).item() for g in task_flat_grads]
        
        self.telemetry_data = {
            "conflict": {},
            "cosine_before": {},
            "cosine_after": {},
            "weights": weights if weights is not None else [1.0, 1.0, 1.0]
        }
        
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                if i < len(task_names) and j < len(task_names):
                    dot = torch.dot(task_flat_grads[i], task_flat_grads[j]).item()
                    cos = dot / (raw_norms[i] * raw_norms[j] + 1e-8)
                    key = f"{task_names[i]},{task_names[j]}"
                    self.telemetry_data["conflict"][key] = 1.0 if dot < 0 else 0.0
                    self.telemetry_data["cosine_before"][key] = cos
            
        # 2. Cắt gọt độ lớn (Gradient Clipping) trên từng task riêng biệt
        scale = scaler.get_scale() if scaler is not None else 1.0
        if self.max_norm is not None and self.max_norm > 0:
            for i in range(num_tasks):
                scaled_norm = task_flat_grads[i].norm(2)
                unscaled_norm = scaled_norm / scale
                if not torch.isfinite(unscaled_norm):
                    continue
                if unscaled_norm > self.max_norm:
                    task_flat_grads[i] = task_flat_grads[i] * (self.max_norm / (unscaled_norm + 1e-8))

        # 3. Phẫu thuật Gradient (PCGrad Projection) - Hỗ trợ Asymmetric Gating
        projected_flat_grads = []
        for i in range(num_tasks):
            g_i = task_flat_grads[i].clone()
            other_indices = list(range(num_tasks))
            other_indices.remove(i)
            random.shuffle(other_indices)

            for j in other_indices:
                g_j = task_flat_grads[j]
                dot_prod = torch.dot(g_i, g_j)
                if dot_prod < 0:
                    # Nếu chạy bất đối xứng và task i có mức ưu tiên cao hơn task j
                    # thì bỏ qua việc chiếu g_i (bảo toàn hướng đi của task chính)
                    if asymmetric and weights is not None and i < len(weights) and j < len(weights):
                        if weights[i] >= weights[j]:
                            continue
                            
                    g_j_norm_sq = torch.dot(g_j, g_j) + 1e-8
                    g_i -= (dot_prod / g_j_norm_sq) * g_j
            projected_flat_grads.append(g_i)

        # Telemetry: Đo cosine similarity sau khi chiếu
        for i in range(num_tasks):
            if i < len(task_names):
                proj_norm = projected_flat_grads[i].norm(2).item()
                dot_before_after = torch.dot(task_flat_grads[i], projected_flat_grads[i]).item()
                cos_after = dot_before_after / (raw_norms[i] * proj_norm + 1e-8)
                self.telemetry_data["cosine_after"][task_names[i]] = cos_after

        # 4. Cộng dồn các gradient đã qua phẫu thuật
        final_flat_grad = torch.stack(projected_flat_grads).sum(dim=0)

        # 5. Khôi phục (Unflatten) gradient
        offset = 0
        final_grads = []
        for p in params:
            numel = p.numel()
            grad_slice = final_flat_grad[offset:offset+numel].view_as(p)
            final_grads.append(grad_slice.clone())
            offset += numel
            
        return final_grads

    def backward(self, losses: List[torch.Tensor], model, scaler=None, encoder_debug_ids=None,
                 weights: List[float] = None, asymmetric: bool = True):
        """Thực hiện lan truyền ngược PCGrad gốc (áp dụng cho toàn bộ tham số)."""
        params = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    params.append(p)

        task_grads = []
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
                
                grads = []
                for p in params:
                    if p.grad is not None:
                        grads.append(p.grad.clone())
                    else:
                        grads.append(torch.zeros_like(p))
                task_grads.append(grads)

        self.optimizer.zero_grad(set_to_none=True)
        if not task_grads:
            return

        # Phân tích gradient
        if encoder_debug_ids is not None:
            # Flatten to task_flat_grads
            task_flat_grads = []
            for grads in task_grads:
                task_flat_grads.append(torch.cat([g.view(-1) for g in grads]))
            self._analyze_encoder_grads(params, task_flat_grads, encoder_debug_ids, scaler)

        final_grads = self._project_grads(task_grads, params, scaler, weights, asymmetric)

        for p, g in zip(params, final_grads):
            p.grad = g.clone()

    def backward_encoder_bypass(self, losses: List[torch.Tensor], model, scaler=None, encoder_debug_ids=None,
                                 weights: List[float] = None, asymmetric: bool = True):
        """
        Thực hiện lan truyền ngược PCGrad tối ưu (Phương án 2):
        - Cô lập Task Paths bằng các chốt chặn detached.
        - Backward qua Task Paths độc lập không retain_graph.
        - Chạy backward qua Shared Decoder và phẫu thuật gradient tại giao diện.
        - Chạy backward qua Encoder đúng 1 lần duy nhất để tối ưu VRAM và tốc độ.
        """
        raw_model = model.module if hasattr(model, "module") else model
        task_leaves = raw_model.decoder.task_leaves  # {"cow": (x_shared_cow, s2_cow, s1_cow), ...}
        
        is_ddp = hasattr(model, "no_sync") and torch.distributed.is_initialized()
        make_context = lambda: model.no_sync() if is_ddp else contextlib.nullcontext()
        
        # 1. Backward 3 nhánh Task Paths độc lập (Không dùng retain_graph)
        model.zero_grad(set_to_none=True)
        with make_context():
            # Task CoW: losses[2]
            if scaler is not None and self.use_amp:
                scaler.scale(losses[2]).backward()
            else:
                losses[2].backward()
                
            # Task LVO: losses[1]
            if scaler is not None and self.use_amp:
                scaler.scale(losses[1]).backward()
            else:
                losses[1].backward()
                
            # Task Lesion: losses[0]
            if scaler is not None and self.use_amp:
                scaler.scale(losses[0]).backward()
            else:
                losses[0].backward()

        # Lấy gradients tại các điểm chốt giao diện của Task Paths
        g_cow = [t.grad for t in task_leaves["cow"]]
        g_lvo = [t.grad for t in task_leaves["lvo"]]
        g_les = [t.grad for t in task_leaves["lesion"]]

        # 2. Backward qua Shared Decoder 3 lần sử dụng grad_tensors
        shared_dec_params = [p for p in raw_model.decoder.shared_bottleneck.parameters() if p.requires_grad] + \
                            [p for p in raw_model.decoder.shared_path.parameters() if p.requires_grad]
                            
        task_dec_grads = []
        rep_grads = []  # list of 3 lists: [ [g_s5, g_s4, g_s3, g_s2, g_s1], ... ]
        
        s5_dec = raw_model.decoder.s5_dec
        s4_dec = raw_model.decoder.s4_dec
        s3_dec = raw_model.decoder.s3_dec
        x_shared = raw_model.decoder.x_shared
        
        with make_context():
            # CoW Shared Path Backward
            for p in shared_dec_params:
                p.grad = None
            s5_dec.grad, s4_dec.grad, s3_dec.grad = None, None, None
            torch.autograd.backward(x_shared, grad_tensors=g_cow[0], retain_graph=True)
            task_dec_grads.append([p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in shared_dec_params])
            rep_grads.append([
                s5_dec.grad.clone() if s5_dec.grad is not None else torch.zeros_like(s5_dec),
                s4_dec.grad.clone() if s4_dec.grad is not None else torch.zeros_like(s4_dec),
                s3_dec.grad.clone() if s3_dec.grad is not None else torch.zeros_like(s3_dec),
                g_cow[1].clone() if g_cow[1] is not None else torch.zeros_like(task_leaves["cow"][1]),
                g_cow[2].clone() if g_cow[2] is not None else torch.zeros_like(task_leaves["cow"][2])
            ])
            
            # LVO Shared Path Backward
            for p in shared_dec_params:
                p.grad = None
            s5_dec.grad, s4_dec.grad, s3_dec.grad = None, None, None
            torch.autograd.backward(x_shared, grad_tensors=g_lvo[0], retain_graph=True)
            task_dec_grads.append([p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in shared_dec_params])
            rep_grads.append([
                s5_dec.grad.clone() if s5_dec.grad is not None else torch.zeros_like(s5_dec),
                s4_dec.grad.clone() if s4_dec.grad is not None else torch.zeros_like(s4_dec),
                s3_dec.grad.clone() if s3_dec.grad is not None else torch.zeros_like(s3_dec),
                g_lvo[1].clone() if g_lvo[1] is not None else torch.zeros_like(task_leaves["lvo"][1]),
                g_lvo[2].clone() if g_lvo[2] is not None else torch.zeros_like(task_leaves["lvo"][2])
            ])
            
            # Lesion Shared Path Backward
            for p in shared_dec_params:
                p.grad = None
            s5_dec.grad, s4_dec.grad, s3_dec.grad = None, None, None
            torch.autograd.backward(x_shared, grad_tensors=g_les[0], retain_graph=False) # Giải phóng đồ thị Shared Decoder!
            task_dec_grads.append([p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in shared_dec_params])
            rep_grads.append([
                s5_dec.grad.clone() if s5_dec.grad is not None else torch.zeros_like(s5_dec),
                s4_dec.grad.clone() if s4_dec.grad is not None else torch.zeros_like(s4_dec),
                s3_dec.grad.clone() if s3_dec.grad is not None else torch.zeros_like(s3_dec),
                g_les[1].clone() if g_les[1] is not None else torch.zeros_like(task_leaves["lesion"][1]),
                g_les[2].clone() if g_les[2] is not None else torch.zeros_like(task_leaves["lesion"][2])
            ])

        # 3. Phẫu thuật gradient (PCGrad) trên Shared Decoder
        final_shared_dec_grads = self._project_grads(task_dec_grads, shared_dec_params, scaler, weights, asymmetric)
        for p, g in zip(shared_dec_params, final_shared_dec_grads):
            p.grad = g.clone()

        # 4. Phẫu thuật gradient (PCGrad) trên giao diện skips biểu diễn
        skips_dummy = [
            s5_dec, s4_dec, s3_dec,
            task_leaves["cow"][1], task_leaves["cow"][2]
        ]
        final_rep_grads = self._project_grads(rep_grads, skips_dummy, scaler, weights, asymmetric)

        # 5. Backward duy nhất 1 lần qua Encoder
        s1_orig, s2_orig, s3_orig, s4_orig, s5_orig = raw_model.encoder.saved_skips
        
        # 5. Backward duy nhất 1 lần qua Encoder nếu Encoder đang hoạt động (yêu cầu grad)
        if any(s.requires_grad for s in [s5_orig, s4_orig, s3_orig, s2_orig, s1_orig]):
            with make_context():
                torch.autograd.backward(
                    [s5_orig, s4_orig, s3_orig, s2_orig, s1_orig],
                    grad_tensors=final_rep_grads,
                    retain_graph=False  # Giải phóng đồ thị Encoder!
                )

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
