"""
trainer.py — Vòng lặp huấn luyện chính cho Dual-Encoder Multi-Task UNet

Chiến lược huấn luyện:
    Phase 1 (epoch 0 → freeze_encoder_epochs):
        - Encoder bị freeze → Chỉ Decoder + Heads học
        - Mục đích: Ổn định Decoder trước khi fine-tune Encoder
    Phase 2 (epoch freeze_encoder_epochs → end):
        - Unfreeze Encoder → Fine-tune toàn bộ mạng
        - Encoder LR thấp hơn 10x (Differential LR)

Features:
    - AMP (Automatic Mixed Precision) với GradScaler cho T4
    - Gradient Clipping để tránh LVO loss explosion (weight 10x)
    - DDP-compatible (DistributedDataParallel)
    - Logging per-batch và per-epoch
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Optional

from compile.metrics import compute_all_metrics
from evaluation.visualize import overlay_predictions


class Trainer:
    """
    Quản lý toàn bộ vòng lặp training và validation.
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn,
        optimizer,
        scheduler,
        config: dict,
        device: torch.device,
        rank: int = 0,
        pcgrad=None,          # [PCGrad] PCGrad instance hoặc None
    ):
        """
        Args:
            model:        DualEncoderUNet (hoặc DDP-wrapped)
            train_loader: DataLoader cho train set
            val_loader:   DataLoader cho validation set
            loss_fn:      MultiTaskLoss
            optimizer:    AdamW
            scheduler:    SequentialLR
            config:       Dict từ train.yaml
            device:       torch.device của GPU hiện tại
            rank:         GPU rank (0 = master)
        """
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.loss_fn      = loss_fn
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.config       = config
        self.device       = device
        self.rank         = rank
        self.output_dir   = config.get("output_dir", "outputs")
        self.pcgrad       = pcgrad  # [PCGrad] None nếu disabled

        train_cfg = config["training"]
        self.epochs            = int(train_cfg["epochs"])
        self.amp_enabled       = bool(train_cfg["amp"])
        self.grad_clip_norm    = float(train_cfg["grad_clip_norm"])
        self.freeze_enc_epochs = int(train_cfg["freeze_encoder_epochs"])
        self.log_interval      = int(train_cfg["logging"]["log_interval"])
        self.metric_weights    = config["composite_score"]

        # API mới của PyTorch (tránh FutureWarning)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled)
        self.history = []

    # ── Một epoch train ──────────────────────────────────────────────────────

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss  = 0.0
        n_batches   = 0
        nan_batches = 0
        nan_input_batches = 0
        nan_flag = torch.zeros(1, device=self.device)

        print("\nStarting Epoch %d:" % (epoch + 1))
        for batch_idx, batch in enumerate(self.train_loader):
            inp = batch["input"].to(self.device, non_blocking=True)   # (B, 18, H, W)
            lbl = batch["label"].to(self.device, non_blocking=True)   # (B, 3, H, W)

            # ── Stage 1: Xử lý NaN trong Input Data ─────────────────────────
            # Thay vì skip batch, thay NaN/inf bằng 0 để không mất dữ liệu.
            # Nguyên nhân NaN: kênh Perfusion (CBF, Tmax) có thể chứa NaN
            # do lỗi tiền xử lý (chia 0, scan bị thiếu).
            if not torch.isfinite(inp).all():
                nan_input_batches += 1
                inp = torch.nan_to_num(inp, nan=0.0, posinf=0.0, neginf=0.0)
            # ─────────────────────────────────────────────────────────────────

            self.optimizer.zero_grad(set_to_none=True)

            # Forward với AMP
            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                preds  = self.model(inp)
                losses = self.loss_fn(preds, lbl)

            # ── Stage 2: Kiểm tra NaN trong Loss (AMP overflow) ──────────────
            if not torch.isfinite(losses["total"]):
                nan_batches += 1
                if self.rank == 0 and nan_batches <= 3: # In 3 lần đầu
                    print(f"  [DEBUG] Batch {batch_idx} Loss NaN! "
                          f"L={losses['lesion']:.4f}, LVO={losses['lvo']:.4f}, C={losses['cow']:.4f}")
                self.optimizer.zero_grad(set_to_none=True)
                continue
            # ─────────────────────────────────────────────────────────────────

            # Backward với GradScaler
            self.scaler.scale(losses["total"]).backward()

            # [PCGrad] Override backbone gradient sau backward — chỉ khi enabled
            if self.pcgrad is not None and "task_losses" in losses:
                self.pcgrad.apply(
                    task_losses=losses["task_losses"],
                    scaler=self.scaler,
                )

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["total"].item()
            n_batches  += 1

            # Log mỗi N batch
            if self.rank == 0 and (batch_idx + 1) % self.log_interval == 0:
                print(
                    f"   Epoch {epoch+1} | Batch {batch_idx+1}/{len(self.train_loader)} "
                    f"| Loss: {losses['total']:.4f} "
                    f"(M:{losses['main']:.3f}, A:{losses['aux']:.3f}, LVO:{losses['lvo']:.3f}, B:{losses['boundary']:.3f})",
                    flush=True
                )

        # ─── ĐỒNG BỘ HÓA DDP (ALL-REDUCE) ───
        # Để báo cáo Train Loss chính xác trên toàn bộ hệ thống
        avg_train_loss = total_loss / max(n_batches, 1)
        if dist.is_initialized():
            sync_data = torch.tensor([total_loss, float(n_batches)], device=self.device)
            dist.all_reduce(sync_data, op=dist.ReduceOp.SUM)
            avg_train_loss = sync_data[0].item() / max(sync_data[1].item(), 1)

        if self.rank == 0 and nan_batches > 0:
            print(f"  [WARN] Epoch {epoch+1}: {nan_batches} batches bị NaN/Inf.", flush=True)

        return {"train_loss": avg_train_loss}

    # ── Validation ────────────────────────────────────────────────────────────


    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        self.model.eval()

        total_loss      = 0.0
        sum_dice_lesion = 0.0
        sum_recall_lvo  = 0.0
        sum_dice_cow    = 0.0
        n_batches     = 0
        n_lvo_batches = 0
        
        # Biến để kiểm tra xem đã vẽ ảnh cho epoch này chưa
        visualized = False
        vis_interval = self.config["training"]["logging"].get("visualize_every", 5)
        should_visualize = (epoch % vis_interval == 0) and (self.rank == 0)

        for batch_idx, batch in enumerate(self.val_loader):
            inp = batch["input"].to(self.device, non_blocking=True)
            lbl = batch["label"].to(self.device, non_blocking=True)

            if not torch.isfinite(inp).all():
                continue

            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                preds  = self.model(inp)
                losses = self.loss_fn(preds, lbl)

            pred_ok = True
            for v in preds.values():
                if isinstance(v, torch.Tensor):
                    if not torch.isfinite(v).all():
                        pred_ok = False
                        break
                elif isinstance(v, list):
                    if not all(torch.isfinite(t).all() for t in v if t is not None):
                        pred_ok = False
                        break
            if not pred_ok:
                continue

            total_loss += losses["total"].item()
            metrics = compute_all_metrics(preds, lbl, self.metric_weights)

            sum_dice_lesion += metrics["dice_lesion"]
            sum_dice_cow    += metrics["dice_cow"]
            n_batches += 1

            if metrics["recall_lvo"] >= 0:
                sum_recall_lvo += metrics["recall_lvo"]
                n_lvo_batches  += 1

            # --- Thực hiện vẽ ảnh (chọn ngẫu nhiên 1 mẫu trong batch đầu tiên hợp lệ) ---
            if should_visualize and not visualized:
                import random
                idx = random.randint(0, inp.size(0) - 1)
                
                # Sử dụng đường dẫn tuyệt đối
                vis_base = self.config["training"]["checkpoint"]["dir"]
                vis_dir = os.path.join(self.output_dir, vis_base, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                
                overlay_predictions(
                    sample={"input": inp[idx], "label": lbl[idx], "path": batch["path"][idx]},
                    preds={k: v[idx:idx+1] for k, v in preds.items()},
                    epoch=epoch,
                    save_dir=vis_dir,
                    thresholds=self.metric_weights.get("thresholds", {})
                )
                visualized = True

        avg_loss        = total_loss / max(n_batches, 1)
        avg_dice_lesion = sum_dice_lesion / max(n_batches, 1)
        avg_recall_lvo  = sum_recall_lvo  / max(n_lvo_batches, 1)
        avg_dice_cow    = sum_dice_cow    / max(n_batches, 1)

        # ─── ĐỒNG BỘ HÓA DDP (ALL-REDUCE) ───
        # Để đảm bảo tính minh bạch và chính xác trên toàn bộ tập Validation
        if dist.is_initialized():
            # Gom tất cả chỉ số vào 1 tensor để gửi đi 1 lần cho nhanh
            sync_data = torch.tensor([
                total_loss, sum_dice_lesion, sum_recall_lvo, sum_dice_cow,
                float(n_batches), float(n_lvo_batches)
            ], device=self.device)
            
            # Cộng dồn từ tất cả các GPU
            dist.all_reduce(sync_data, op=dist.ReduceOp.SUM)
            
            # Tính toán lại trung bình trên quy mô toàn hệ thống
            v = sync_data.cpu().numpy()
            avg_loss        = v[0] / max(v[4], 1)
            avg_dice_lesion = v[1] / max(v[4], 1)
            avg_recall_lvo  = v[2] / max(v[5], 1)
            avg_dice_cow    = v[3] / max(v[4], 1)

        # 3. Tính Composite Score
        w = self.metric_weights
        composite = (
            w["dice_lesion_weight"] * avg_dice_lesion +
            w["recall_lvo_weight"]  * avg_recall_lvo  +
            w["dice_cow_weight"]    * avg_dice_cow
        )

        return {
            "val_loss":    avg_loss,
            "dice_lesion": avg_dice_lesion,
            "recall_lvo":  avg_recall_lvo,
            "dice_cow":    avg_dice_cow,
            "composite":   composite,
        }


    # ── Vòng lặp chính ───────────────────────────────────────────────────────

    def fit(self, early_stopping=None, checkpoint=None):
        """
        Chạy toàn bộ training loop.

        Args:
            early_stopping: EarlyStopping instance (optional)
            checkpoint:     ModelCheckpoint instance (optional)
        """
        # Unwrap raw model (để gọi freeze/unfreeze)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        # Phase 1: Freeze encoder
        raw_model.freeze_encoders()

        for epoch in range(self.epochs):
            # Unfreeze khi đến epoch chỉ định
            if epoch == self.freeze_enc_epochs:
                raw_model.unfreeze_encoders()
                # [PCGrad] Refresh danh sách params sau unfreeze (encoder params được active)
                if self.pcgrad is not None:
                    self.pcgrad.refresh_params()

            # Set epoch cho DistributedSampler
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            train_metrics = self.train_one_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch + 1)

            # Log kết quả epoch
            if self.rank == 0:
                curr_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"--------------------------------------------------"
                    f"\n=> | [Epoch {epoch+1:03d}/{self.epochs}] | LR: {curr_lr:.2e} "
                    f"\n   | Dice_L: {val_metrics['dice_lesion']:.4f} | "
                    f"Recall_LVO: {val_metrics['recall_lvo']:.4f} | "
                    f"Dice_C: {val_metrics['dice_cow']:.4f}"
                    f"\n   | Loss(T/V): {train_metrics['train_loss']:.4f}/{val_metrics['val_loss']:.4f} | "
                    f"Composite: {val_metrics['composite']:.4f}",
                    flush=True
                )

            # Lưu history
            self.history.append({**train_metrics, **val_metrics, "epoch": epoch + 1})

            # Checkpoint (Chỉ lưu sau khi đạt start_epoch)
            if checkpoint is not None and self.rank == 0:
                start_ckpt = self.config["training"]["checkpoint"].get("start_epoch", 1)
                if (epoch + 1) >= start_ckpt:
                    checkpoint.update(self.model, self.optimizer, epoch + 1, val_metrics)

            # Early stopping (Chỉ tính patience sau khi đạt start_epoch)
            if early_stopping is not None:
                start_es = self.config["training"]["early_stopping"].get("start_epoch", 1)
                if (epoch + 1) >= start_es:
                    if early_stopping(val_metrics["composite"]):
                        if self.rank == 0:
                            print(f"[Trainer] Early stopping tại epoch {epoch+1}", flush=True)
                        break
            
            # Cập nhật LR sau mỗi epoch (Sau optimizer.step đã chạy trong train_one_epoch)
            self.scheduler.step()

        return self.history
