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
        self.output_dir   = config.get("output_dir", "outputs") # Thêm dòng này

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
                losses = self.loss_fn(preds, lbl, epoch=epoch, batch_idx=batch_idx)

            # ── [DEBUG] Monitor Gating Signal (Khả năng 1) ──────────────────
            # Chỉ in định kỳ để tránh làm loãng log
            if batch_idx % self.log_interval == 0 and self.rank == 0:
                g_maps = preds.get("guidance_maps", {})
                v_map = g_maps.get("v_guidance") # (B, 1, H, W)
                if v_map is not None:
                    m_val = v_map.mean().item()
                    s_val = v_map.std().item()
                    print(f"    [GATE_DEBUG] Epoch {epoch+1} Batch {batch_idx}: Mean={m_val:.3f}, Std={s_val:.3f}")
                    if m_val > 0.45 and m_val < 0.55 and s_val < 0.05:
                        print("    [WARN] Gate is currently uniform. Vascular Guidance not yet effective.")
            # ─────────────────────────────────────────────────────────────────

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

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)

            # ── [DEBUG] Monitor Branch Gradients (Khả năng 3) ───────────────
            # Theo dõi xem LVO branch có nhận đủ gradient không
            if batch_idx % self.log_interval == 0 and self.rank == 0:
                grad_norms = {"lesion": 0.0, "lvo": 0.0, "cow": 0.0}
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        gnorm = param.grad.detach().norm(2).item()
                        if "lesion" in name.lower(): grad_norms["lesion"] += gnorm
                        elif "lvo" in name.lower():   grad_norms["lvo"]    += gnorm
                        elif "cow" in name.lower():   grad_norms["cow"]    += gnorm
                
                print(f"    [GRAD_DEBUG] B{batch_idx} Norms: LVO={grad_norms['lvo']:.4f}, Lesion={grad_norms['lesion']:.4f}, CoW={grad_norms['cow']:.4f}")
                if grad_norms["lvo"] < 0.0001 and epoch > 5:
                    print("    [WARN] LVO branch gradient is extremely low. Potential Vanishing Gradient.")
            # ───────────────────────────────────────────────────────────────

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["total"].item()
            n_batches  += 1

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
        
        sum_l_L_tv, sum_l_L_hd = 0.0, 0.0
        sum_l_C_tv, sum_l_C_cl = 0.0, 0.0
        
        n_batches     = 0
        n_lvo_batches = 0
        
        # Biến tích lũy cho thi đua
        sum_l_L, sum_l_V, sum_l_C = 0.0, 0.0, 0.0
        sum_p_lvo, sum_sigma_lvo = 0.0, 0.0
        
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
                losses = self.loss_fn(preds, lbl, epoch=epoch)

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
            sum_l_L += losses.get("l_lesion", 0.0)
            sum_l_V += losses.get("l_lvo", 0.0)
            sum_l_C += losses.get("l_cow", 0.0)
            
            # Theo dõi hỏa lực và độ bất định
            sum_p_lvo += losses.get("p_lvo", 1.0)
            sum_sigma_lvo += losses.get("sigma_lvo", 1.0)
            
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
                    preds={k: v[idx:idx+1] for k, v in preds.items() if isinstance(v, torch.Tensor)},
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
        if dist.is_initialized():
            sync_data = torch.tensor([
                total_loss, sum_dice_lesion, sum_recall_lvo, sum_dice_cow,
                sum_l_L_tv, sum_l_L_hd, sum_l_C_tv, sum_l_C_cl,
                float(n_batches), float(n_lvo_batches)
            ], device=self.device)
            dist.all_reduce(sync_data, op=dist.ReduceOp.SUM)
            
            v = sync_data.cpu().numpy()
            avg_loss        = v[0] / max(v[8], 1)
            avg_dice_lesion = v[1] / max(v[8], 1)
            avg_recall_lvo  = v[2] / max(v[9], 1)
            avg_dice_cow    = v[3] / max(v[8], 1)
            
            avg_l_L_tv = v[4] / max(v[8], 1)
            avg_l_L_hd = v[5] / max(v[8], 1)
            avg_l_C_tv = v[6] / max(v[8], 1)
            avg_l_C_cl = v[7] / max(v[8], 1)
        else:
            avg_loss        = total_loss / max(n_batches, 1)
            avg_dice_lesion = sum_dice_lesion / max(n_batches, 1)
            avg_recall_lvo  = sum_recall_lvo  / max(n_lvo_batches, 1)
            avg_dice_cow    = sum_dice_cow    / max(n_batches, 1)
            avg_l_L_tv = sum_l_L_tv / max(n_batches, 1)
            avg_l_L_hd = sum_l_L_hd / max(n_batches, 1)
            avg_l_C_tv = sum_l_C_tv / max(n_batches, 1)
            avg_l_C_cl = sum_l_C_cl / max(n_batches, 1)

        # 3. Tính Composite Score
        w = self.metric_weights
        composite = (
            w["dice_lesion_weight"] * avg_dice_lesion +
            w["recall_lvo_weight"]  * avg_recall_lvo  +
            w["dice_cow_weight"]    * avg_dice_cow
        )

        # Lấy giá trị Sigmas từ loss_fn nếu có
        import math
        sigmas = {"sigma_lesion": 1.0, "sigma_lvo": 1.0, "sigma_cow": 1.0}
        if hasattr(self.loss_fn, "log_vars"):
            sigmas["sigma_lesion"] = math.exp(self.loss_fn.log_vars["lesion"].item() / 2.0)
            sigmas["sigma_lvo"]    = math.exp(self.loss_fn.log_vars["lvo"].item() / 2.0)
            sigmas["sigma_cow"]    = math.exp(self.loss_fn.log_vars["cow"].item() / 2.0)

        return {
            "val_loss":    avg_loss,
            "dice_lesion": avg_dice_lesion,
            "recall_lvo":  avg_recall_lvo,
            "dice_cow":    avg_dice_cow,
            "composite":   composite,
            "l_L_tv":      avg_l_L_tv,
            "l_L_hd":      avg_l_L_hd,
            "l_C_tv":      avg_l_C_tv,
            "l_C_cl":      avg_l_C_cl,
            **sigmas
        }


    # ── Vòng lặp chính ───────────────────────────────────────────────────────

    def load_checkpoint(self, checkpoint_path: str):
        """Nạp lại trạng thái huấn luyện từ file .pt."""
        if not os.path.exists(checkpoint_path):
            if self.rank == 0:
                print(f"[Trainer] KHÔNG tìm thấy file checkpoint: {checkpoint_path}")
            return 0

        if self.rank == 0:
            print(f"[Trainer] Đang nạp checkpoint từ: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Unwrap model nếu đang dùng DDP
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_model.load_state_dict(checkpoint["model"])
        
        if "optimizer" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        if "scheduler" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            
        if "history" in checkpoint:
            self.history = checkpoint["history"]
            
        start_epoch = checkpoint.get("epoch", 0)
        return start_epoch

    def fit(self, early_stopping=None, checkpoint=None, start_epoch: int = 0):
        """
        Args:
            early_stopping: EarlyStopping instance (optional)
            checkpoint:     ModelCheckpoint instance (optional)
            start_epoch:    Epoch bắt đầu (mặc định 0)
        """
        # Unwrap raw model (để gọi freeze/unfreeze)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        # Phase 1: Mặc định freeze trước
        raw_model.freeze_encoders()

        for epoch in range(start_epoch, self.epochs):
            # Unfreeze khi đến epoch chỉ định
            if epoch == self.freeze_enc_epochs:
                raw_model.unfreeze_encoders()

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
                    f"--------------------------------------------------------------------------------"
                    f"\n=> | [Epoch {epoch+1:03d}/{self.epochs}] | LR: {curr_lr:.2e} | Composite: {val_metrics['composite']:.4f}"
                    f"\n   | [VAL METRICS] Dice_Lesion: {val_metrics['dice_lesion']:.4f} | Recall_LVO: {val_metrics['recall_lvo']:.4f} | Dice_CoW: {val_metrics['dice_cow']:.4f}"
                    f"\n   | [TRAIN LOSS ] Avg_Total: {train_metrics['train_loss']:.4f}"
                    f"\n   | [COMPETITION] P_LVO: {val_metrics.get('p_lvo', 1.0):.2f} | Sigma_LVO: {val_metrics.get('sigma_lvo', 1.0):.2f}"
                    f"\n--------------------------------------------------------------------------------",
                    flush=True
                )

            # Lưu history
            self.history.append({**train_metrics, **val_metrics, "epoch": epoch + 1})

            # Checkpoint (Chỉ lưu sau khi đạt start_epoch)
            if checkpoint is not None and self.rank == 0:
                start_ckpt = self.config["training"]["checkpoint"].get("start_epoch", 1)
                if (epoch + 1) >= start_ckpt:
                    checkpoint.update(
                        self.model, 
                        self.optimizer, 
                        epoch + 1, 
                        val_metrics,
                        scheduler=self.scheduler,
                        history=self.history
                    )

            # Early stopping (Chỉ tính patience sau khi đạt start_epoch)
            if early_stopping is not None:
                start_es = self.config["training"]["early_stopping"].get("start_epoch", 1)
                if (epoch + 1) >= start_es:
                    if early_stopping(val_metrics["composite"]):
                        if self.rank == 0:
                            print(f"[Trainer] Early stopping tại epoch {epoch+1}", flush=True)
                        break
            
            # Cập nhật LR sau mỗi epoch
            self.scheduler.step()

            # [QUAN TRỌNG] Cập nhật thống kê thi đua (DWA) cho epoch tiếp theo
            if hasattr(self.loss_fn, "update_epoch_stats"):
                self.loss_fn.update_epoch_stats()

        return self.history
