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

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Optional

from compile.metrics import compute_all_metrics


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
        total_loss = 0.0
        n_batches  = 0
        nan_batches = 0
        # Flag dùng để đồng bộ quyết định NaN giữa các rank
        nan_flag = torch.zeros(1, device=self.device)

        for batch_idx, batch in enumerate(self.train_loader):
            inp = batch["input"].to(self.device, non_blocking=True)   # (B, 18, H, W)
            lbl = batch["label"].to(self.device, non_blocking=True)   # (B, 3, H, W)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward với AMP
            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                preds  = self.model(inp)
                losses = self.loss_fn(preds, lbl)

            # ── NaN Guard ────────────────────────────────────────────────────────
            # Bắt buộc đồng bộ quyết định giữa các rank qua all_reduce.
            # Nếu bỏ qua bằng `continue` độc lập, mỗi rank sẽ có số lượng
            # ALLREDUCE khác nhau → NCCL deadlock (timeout sau 10 phút).
            nan_flag.fill_(0.0)
            if not torch.isfinite(losses["total"]):
                nan_flag.fill_(1.0)
            # Tất cả rank phải gọi all_reduce — nhận MAX (1 rank NaN → tất cả skip)
            dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)

            if nan_flag.item() > 0:
                nan_batches += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue
            # ──────────────────────────────────────────────────────────────────

            # Backward với GradScaler
            self.scaler.scale(losses["total"]).backward()

            # Gradient clipping — bảo vệ khỏi LVO loss explosion
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["total"].item()
            n_batches  += 1

            # Log mỗi N batch — flush=True để output hiển thị ngay trong Kaggle Notebook
            if self.rank == 0 and (batch_idx + 1) % self.log_interval == 0:
                print(
                    f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(self.train_loader)} "
                    f"| Loss: {losses['total']:.4f} "
                    f"(L={losses['lesion']:.4f}, LVO={losses['lvo']:.4f}, C={losses['cow']:.4f})",
                    flush=True
                )

        if self.rank == 0 and nan_batches > 0:
            print(f"  [WARN] Epoch {epoch+1}: {nan_batches}/{len(self.train_loader)} batches bị NaN (bỏ qua).", flush=True)

        return {"train_loss": total_loss / max(n_batches, 1)}

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()

        sum_metrics = {
            "dice_lesion": 0.0, "recall_lvo": 0.0, "dice_cow": 0.0, "composite": 0.0
        }
        n_batches = 0

        for batch in self.val_loader:
            inp = batch["input"].to(self.device, non_blocking=True)
            lbl = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                preds = self.model(inp)

            metrics = compute_all_metrics(preds, lbl, self.metric_weights)
            for k in sum_metrics:
                sum_metrics[k] += metrics[k]
            n_batches += 1

        # Mean qua tất cả batch
        avg = {k: v / max(n_batches, 1) for k, v in sum_metrics.items()}
        return avg

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

            # Set epoch cho DistributedSampler
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            train_metrics = self.train_one_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Scheduler step
            self.scheduler.step()

            # Log kết quả epoch
            if self.rank == 0:
                print(
                    f"\n[Epoch {epoch+1:03d}/{self.epochs}] "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Dice_L: {val_metrics['dice_lesion']:.4f} | "
                    f"Recall_LVO: {val_metrics['recall_lvo']:.4f} | "
                    f"Dice_CoW: {val_metrics['dice_cow']:.4f} | "
                    f"Composite: {val_metrics['composite']:.4f}",
                    flush=True
                )

            # Lưu history
            self.history.append({**train_metrics, **val_metrics, "epoch": epoch + 1})

            # Checkpoint
            if checkpoint is not None and self.rank == 0:
                checkpoint.update(self.model, self.optimizer, epoch + 1, val_metrics)

            # Early stopping
            if early_stopping is not None:
                if early_stopping(val_metrics["composite"]):
                    if self.rank == 0:
                        print(f"[Trainer] Early stopping tại epoch {epoch+1}", flush=True)
                    break

        return self.history
