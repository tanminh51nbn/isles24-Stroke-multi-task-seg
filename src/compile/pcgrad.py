"""
pcgrad.py — PCGrad: Gradient Surgery for Multi-Task Learning

Paper: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.

Ý tưởng:
    Khi gradient của Task A và Task B xung đột (dot product < 0) trên shared backbone,
    chiếu gradient của Task A lên mặt phẳng vuông góc với Task B (và ngược lại).
    Điều này loại bỏ phần "đối kháng" mà KHÔNG giảm cường độ học của bất kỳ task nào.

Chiến lược tích hợp AMP (đơn giản và đúng):
    1. scaler.scale(total_loss).backward()        → Tất cả params nhận gradient bình thường
    2. Lưu gradient backbone vào bộ nhớ           → Dùng để tính PCGrad
    3. Zero backbone.grad                          → Reset để viết lại gradient PCGrad
    4. Tính PCGrad (project + combine)             → Gradient tổng hợp không xung đột
    5. Gán PCGrad gradient vào backbone.grad       → Head params giữ nguyên gradient từ B1
    6. scaler.unscale_() → clip → step → update   → Như bình thường

Phạm vi:
    Chỉ backbone params (encoder + decoder). Heads độc lập, không conflict.
"""

import torch
from typing import List


class PCGrad:
    """
    Gradient Surgery cho Multi-Task Learning.

    Cách tích hợp vào train loop:

        # B1: Backward bình thường (để head params nhận gradient)
        self.scaler.scale(losses["total"]).backward()

        # B2: PCGrad override backbone gradient
        if self.pcgrad is not None:
            self.pcgrad.apply(losses["task_losses"], self.scaler)

        # B3: Unscale, clip, step như cũ
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
    """

    def __init__(self, backbone_params: list):
        """
        Args:
            backbone_params: List nn.Parameter của backbone (encoder + decoder).
                             KHÔNG bao gồm head parameters.
        """
        self._all_backbone_params = backbone_params
        self.backbone_params = [p for p in backbone_params if p.requires_grad]
        print(f"[PCGrad] Initialized | Active backbone params: {len(self.backbone_params)}")

    def refresh_params(self):
        """
        Cập nhật danh sách params sau khi encoder unfreeze.
        Gọi trong trainer.py khi epoch == freeze_enc_epochs.
        """
        self.backbone_params = [p for p in self._all_backbone_params if p.requires_grad]
        print(f"[PCGrad] Refreshed | Active backbone params: {len(self.backbone_params)}")

    @torch.no_grad()
    def _project_conflicting(self, flat_grads: List[torch.Tensor]) -> torch.Tensor:
        """
        Chiếu các gradient xung đột giữa các task.

        Với mỗi cặp (i, j): nếu dot(gi, gj) < 0 → gi đang "chống lại" gj
            gi_new = gi - (dot(gi, gj) / ||gj||²) × gj

        Args:
            flat_grads: List[Tensor 1D float32] — gradient mỗi task, đã flatten

        Returns:
            Tensor 1D — tổng gradient sau khi chiếu (sẵn sàng reshape về param shape)
        """
        projected = [g.clone() for g in flat_grads]

        for i in range(len(projected)):
            for j, gj_ref in enumerate(flat_grads):
                if i == j:
                    continue
                dot = torch.dot(projected[i], gj_ref)
                if dot < 0:  # Conflict!
                    norm_sq = torch.dot(gj_ref, gj_ref).clamp(min=1e-12)
                    projected[i] = projected[i] - (dot / norm_sq) * gj_ref

        return sum(projected)

    def apply(
        self,
        task_losses: List[torch.Tensor],
        scaler: "torch.amp.GradScaler",
    ):
        """
        Tính gradient PCGrad và override gradient backbone.

        Gọi SAU khi scaler.scale(total_loss).backward() đã chạy xong,
        và TRƯỚC khi scaler.unscale_().

        Quy trình:
            1. Lấy scale_factor từ scaler (gradient hiện tại còn scaled)
            2. Với mỗi task loss, tính gradient backbone (không qua scaler — tự scale thủ công)
            3. Project conflicting gradients
            4. Zero backbone.grad → gán PCGrad gradient thay thế
               (head params giữ nguyên gradient từ backward trước đó)

        Args:
            task_losses: losses["task_losses"] — List[Tensor], 3 loss chưa weighted sum
            scaler:      GradScaler — để lấy scale_factor hiện tại
        """
        scale = scaler.get_scale()
        task_flat_grads = []

        # ── Bước 1: Thu thập gradient backbone của từng task riêng lẻ ──────────
        for task_idx, task_loss in enumerate(task_losses):
            # Backward task này với retain_graph=True (không xóa graph giữa chừng)
            # Scale thủ công để nhất quán với gradient từ backward(total_loss) đã scaled
            (task_loss * scale).backward(retain_graph=True)

            # Thu thập gradient backbone (vẫn đang ở dạng scaled)
            flat_parts = []
            for p in self.backbone_params:
                if p.grad is not None:
                    flat_parts.append(p.grad.detach().float().flatten())
                else:
                    flat_parts.append(
                        torch.zeros(p.numel(), dtype=torch.float32, device=p.device)
                    )
            task_flat_grads.append(torch.cat(flat_parts))

            # Reset backbone.grad sau mỗi task để task kế tiếp bắt đầu sạch
            for p in self.backbone_params:
                if p.grad is not None:
                    p.grad.zero_()

        # ── Bước 2: Project và tổng hợp ────────────────────────────────────────
        combined_flat = self._project_conflicting(task_flat_grads)

        # ── Bước 3: Override backbone.grad bằng PCGrad gradient ─────────────────
        # Lưu ý: head params vẫn giữ gradient từ backward(total_loss) ở bước trước
        # vì ta chỉ gán lại backbone params ở đây.
        offset = 0
        for p in self.backbone_params:
            numel = p.numel()
            pcgrad_slice = combined_flat[offset: offset + numel]
            p.grad = pcgrad_slice.reshape(p.shape).to(dtype=p.dtype)
            offset += numel
