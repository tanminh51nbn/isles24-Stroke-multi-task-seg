import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image

try:
    from accelerate import Accelerator
except ImportError:
    raise ImportError("pip install accelerate is required.")

logger = logging.getLogger(__name__)

# --- Helper Metrics ---
def _dice_score(logits, target, mask, threshold=0.5):
    pred = (torch.sigmoid(logits) > threshold).float()
    pred = pred * mask
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return inter, union

def _recall_score(logits, target, mask, threshold=0.5):
    pred = (torch.sigmoid(logits) > threshold).float()
    pred = pred * mask
    tp = (pred * target).sum()
    tp_fn = target.sum()
    return tp, tp_fn

# --- Trainer ---
class MultiTaskTrainer:
    TASK_NAMES = ("lesion", "lvo", "cow")

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader,
        val_loader,
        cfg: dict,
    ):
        self.cfg = cfg
        train_cfg = cfg["training"]
        
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

        # 1. Initialize Accelerate
        self.accelerator = Accelerator(
            mixed_precision="fp16" if train_cfg.get("amp", True) else "no",
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
            kwargs_handlers=[ddp_kwargs]
        )

        # 2. Prepare with Accelerate
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
        self.criterion = criterion # No parameters, no need to prepare

        # 3. Config
        self.epochs = train_cfg["epochs"]
        self.grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
        self.log_interval = train_cfg.get("logging", {}).get("log_interval", 10)
        self.interactive = train_cfg.get("logging", {}).get("interactive", False)
        
        # Freeze config
        self.freeze_epochs = train_cfg.get("freeze_encoder_epochs", 5)

        # Early stopping & Checkpoints
        self.es_patience = train_cfg.get("early_stopping", {}).get("patience", 15)
        self.ckpt_dir = Path(train_cfg.get("checkpoint", {}).get("dir", "checkpoints"))
        
        # Tracking
        self.best_composite = 0.0
        self.patience_counter = 0

        if self.accelerator.is_main_process:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            (self.ckpt_dir / "visualizations").mkdir(parents=True, exist_ok=True)
            logger.info(f"Trainer initialized (Accelerate). GPUs: {self.accelerator.num_processes}")

    def _freeze_encoder(self):
        """Freezes the encoder weights (ResNet50)."""
        module = self.model.module if hasattr(self.model, "module") else self.model
        for param in module.encoder.parameters():
            param.requires_grad = False
            
    def _unfreeze_encoder(self):
        """Unfreezes the encoder weights."""
        module = self.model.module if hasattr(self.model, "module") else self.model
        for param in module.encoder.parameters():
            param.requires_grad = True

    def train(self):
        logger.info("=" * 60)
        logger.info("  ISLES'24 Multi-Task Training (Accelerate)")
        logger.info("=" * 60)

        for epoch in range(self.epochs):
            # Freeze scheduling
            if epoch < self.freeze_epochs:
                self._freeze_encoder()
                if epoch == 0 and self.accelerator.is_main_process:
                    logger.info("❄️ Encoder frozen for initial epochs.")
            elif epoch == self.freeze_epochs:
                self._unfreeze_encoder()
                if self.accelerator.is_main_process:
                    logger.info("🔥 Encoder unfrozen for full fine-tuning.")

            start_t = time.time()

            if self.accelerator.is_main_process:
                print(f"\n🚀 Epoch {epoch+1:02d} started...", flush=True)

            train_loss, train_tasks, grad_norm = self._train_one_epoch(epoch)
            
            if self.accelerator.is_main_process:
                print(f"🧪 Running Validation...", flush=True)
                
            val_loss, val_metrics = self._validate(epoch)

            # Visualization every N epochs
            visualize_every = self.cfg["training"].get("logging", {}).get("visualize_every", 5)
            if (epoch + 1) % visualize_every == 0 and self.accelerator.is_main_process:
                self._visualize_results(epoch)

            elapsed = time.time() - start_t
            self.scheduler.step()

            # Logging & Checkpointing (Main Process)
            if self.accelerator.is_main_process:
                comp = val_metrics.get("composite_score", 0.0)
                lr = self.optimizer.param_groups[-1]["lr"]
                
                # Lấy chi tiết loss từng task
                t_tasks = train_tasks # dict từ _train_one_epoch
                v_tasks = val_metrics.get("task_losses", {})
                
                # Thiết kế bảng 5 hàng cực kỳ chi tiết
                msg = (
                    f"\n ┌─────────────────────────┬─────────────────────────┬─────────────────────────┐\n"
                    f" │ Epoch: {epoch+1:02d}/{self.epochs:02d}          │ Train Loss: {train_loss:9.4f} │ Val Loss: {val_loss:9.4f}   │\n"
                    f" ├─────────────────────────┼─────────────────────────┼─────────────────────────┤\n"
                    f" │ Train Task Loss Detail: │ Lesion: {t_tasks['lesion']:11.4f} │ LVO: {t_tasks['lvo']:14.4f} │\n"
                    f" │                         │ CoW:    {t_tasks['cow']:11.4f} │                         │\n"
                    f" ├─────────────────────────┼─────────────────────────┼─────────────────────────┤\n"
                    f" │ Val Task Loss Detail:   │ Lesion: {v_tasks.get('lesion',0):11.4f} │ LVO: {v_tasks.get('lvo',0):14.4f} │\n"
                    f" │                         │ CoW:    {v_tasks.get('cow',0):11.4f} │                         │\n"
                    f" ├─────────────────────────┼─────────────────────────┼─────────────────────────┤\n"
                    f" │ Lesion Dice: {val_metrics.get('dice_lesion', 0):8.4f} │ CoW Dice: {val_metrics.get('dice_cow', 0):8.4f}    │ LVO Recall: {val_metrics.get('recall_lvo', 0):8.4f}  │\n"
                    f" ├─────────────────────────┼─────────────────────────┼─────────────────────────┤\n"
                    f" │ Score: {comp:14.4f} │ LVO Pos: {val_metrics.get('lvo_pos_rate', 0)*100:6.2f}%    │ LR: {lr:.1e}           │\n"
                    f" └─────────────────────────┴─────────────────────────┴─────────────────────────┘"
                )
                print(msg, flush=True)

                # Sanity Check Message
                lvo_pos_rate = val_metrics.get('lvo_pos_rate', 0) * 100
                status = "✅ STABLE" if lvo_pos_rate < 5.0 else "⚠️ OVER-PREDICTING"
                print(f" [Sanity Check] LVO Predicted Pixels: {lvo_pos_rate:.3f}% ({status})", flush=True)

                # Checkpoint
                if comp > self.best_composite:
                    self.best_composite = comp
                    self.patience_counter = 0
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                        "composite": comp
                    }, self.ckpt_dir / "best_model.pth")
                    print(f"  --> Saved new best model (Composite: {comp:.4f})", flush=True)
                else:
                    self.patience_counter += 1

                # Early Stopping
                if self.patience_counter >= self.es_patience:
                    print(f"Early stopping triggered after {epoch} epochs.", flush=True)
                    break
                    
        # Synchronize across processes before exiting
        self.accelerator.wait_for_everyone()

    def _train_one_epoch(self, epoch: int) -> Tuple[float, Dict[str, float], float]:
        self.model.train()
        running_loss = 0.0
        task_loss_sums = {t: 0.0 for t in self.TASK_NAMES}
        total_grad_norm = 0.0
        grad_steps = 0
        n_batches = 0

        loader = self.train_loader
        if self.interactive:
            loader = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch+1:02d} [Train]", 
                disable=not self.accelerator.is_local_main_process
            )

        for step, (x, y, brain_mask) in enumerate(loader):
            with self.accelerator.accumulate(self.model):
                with self.accelerator.autocast():
                    preds = self.model(x)
                    preds = list(preds)
                    
                    # Apply mask (background to -1e9)
                    for i in range(len(preds)):
                        preds[i] = preds[i] * brain_mask + (-1e9) * (1 - brain_mask)
                        
                    loss, loss_dict = self.criterion(preds, y)

                self.accelerator.backward(loss)

                # Gradient clipping
                if self.accelerator.sync_gradients:
                    current_grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    total_grad_norm += current_grad_norm.item()
                    grad_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()

            real_loss = loss.item()
            running_loss += real_loss
            n_batches += 1
            for t in self.TASK_NAMES:
                task_loss_sums[t] += loss_dict[t]

        # Sync loss and tasks across processes
        avg_loss_t = torch.tensor([running_loss], device=self.accelerator.device)
        total_loss = self.accelerator.reduce(avg_loss_t, reduction="sum").item()
        
        n_batches_t = torch.tensor([float(n_batches)], device=self.accelerator.device)
        total_batches = self.accelerator.reduce(n_batches_t, reduction="sum").item()
        
        avg_loss = total_loss / max(total_batches, 1)
        
        # Sync task losses
        task_loss_t = torch.tensor([task_loss_sums[t] for t in self.TASK_NAMES], device=self.accelerator.device)
        total_task_losses = self.accelerator.reduce(task_loss_t, reduction="sum")
        avg_tasks = {t: (total_task_losses[i] / max(total_batches, 1)).item() for i, t in enumerate(self.TASK_NAMES)}
        
        avg_grad = total_grad_norm / max(grad_steps, 1)
        
        return avg_loss, avg_tasks, avg_grad

    @torch.no_grad()
    def _validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        running_loss = 0.0
        n_batches = 0
        task_loss_sums = {t: 0.0 for t in self.TASK_NAMES}
        accum = torch.zeros(8, device=self.accelerator.device) # Increased to 8 for PosRate

        loader = self.val_loader
        if self.interactive:
            loader = tqdm(
                self.val_loader, 
                desc=f"Epoch {epoch+1:02d} [Val]  ", 
                disable=not self.accelerator.is_local_main_process
            )

        for x, y, brain_mask in loader:
            with self.accelerator.autocast():
                preds = self.model(x)
                preds = list(preds)
                for i in range(len(preds)):
                    preds[i] = preds[i] * brain_mask + (-1e9) * (1 - brain_mask)
                loss, loss_dict = self.criterion(preds, y)

            running_loss += loss.item()
            n_batches += 1
            for t in self.TASK_NAMES:
                task_loss_sums[t] += loss_dict[t]
            
            # 1. Lesion Dice
            inter_l, union_l = _dice_score(preds[0], y[:, 0:1], brain_mask)
            accum[0] += inter_l
            accum[1] += union_l
            
            # 2. LVO Recall
            tp, tp_fn = _recall_score(preds[1], y[:, 1:2], brain_mask)
            accum[2] += tp
            accum[3] += tp_fn
            
            # 3. CoW Dice
            inter_c, union_c = _dice_score(preds[2], y[:, 2:3], brain_mask)
            accum[4] += inter_c
            accum[5] += union_c
            
            # 4. LVO Positive Rate (Sanity Check)
            lvo_pos = (torch.sigmoid(preds[1]) > 0.5).float().sum()
            total_px = brain_mask.sum()
            accum[6] += lvo_pos
            accum[7] += total_px

        # Global sync
        accum = self.accelerator.reduce(accum, reduction="sum")
        
        avg_loss_t = torch.tensor([running_loss], device=self.accelerator.device)
        total_loss = self.accelerator.reduce(avg_loss_t, reduction="sum").item()
        
        n_batches_t = torch.tensor([float(n_batches)], device=self.accelerator.device)
        total_batches = self.accelerator.reduce(n_batches_t, reduction="sum").item()
        
        avg_loss = total_loss / max(total_batches, 1)
        
        # Sync task losses
        task_loss_t = torch.tensor([task_loss_sums[t] for t in self.TASK_NAMES], device=self.accelerator.device)
        total_task_losses = self.accelerator.reduce(task_loss_t, reduction="sum")
        avg_tasks = {t: (total_task_losses[i] / max(total_batches, 1)).item() for i, t in enumerate(self.TASK_NAMES)}

        metrics = {}
        if self.accelerator.is_main_process:
            d_les = (2.0 * accum[0] / (accum[1] + 1e-6)).item()
            r_lvo = (accum[2] / (accum[3] + 1e-6)).item()
            d_cow = (2.0 * accum[4] / (accum[5] + 1e-6)).item()
            comp = 0.4 * d_les + 0.4 * r_lvo + 0.2 * d_cow
            
            metrics = {
                "dice_lesion": d_les,
                "recall_lvo": r_lvo,
                "dice_cow": d_cow,
                "lvo_pos_rate": (accum[6] / (accum[7] + 1e-6)).item(),
                "composite_score": comp,
                "task_losses": avg_tasks
            }
            
        # CRITICAL: Sync composite_score for Early Stopping
        comp_t = torch.tensor([metrics.get("composite_score", 0.0)], device=self.accelerator.device)
        comp_t = self.accelerator.reduce(comp_t, reduction="sum")
        if not self.accelerator.is_main_process:
            metrics["composite_score"] = comp_t.item()
            
        return avg_loss, metrics

    @torch.no_grad()
    def _visualize_results(self, epoch: int):
        """
        Saves overlay visualizations of NCCT, Ground Truth, and Predictions.
        Focuses on slices with LVO or Lesions.
        """
        self.model.eval()
        # Lấy 1 batch từ val_loader
        try:
            x, y, brain_mask = next(iter(self.val_loader))
        except StopIteration:
            return

        with self.accelerator.autocast():
            preds = self.model(x)
            preds = list(preds)
            # Masking
            for i in range(len(preds)):
                preds[i] = preds[i] * brain_mask + (-1e9) * (1 - brain_mask)
            
            probs = [torch.sigmoid(p) for p in preds]

        # Chuyển sang numpy
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        p_np = [pr.cpu().numpy() for pr in probs]
        
        # Diversified Sampling: 1 Empty, 1 LVO+, 1 Lesion+, 1 CoW+
        with torch.no_grad():
            y_sums = y.sum(dim=(2, 3)) # [B, 3]
            indices = []
            
            # 1. Tìm LVO
            lvo_ids = torch.where(y_sums[:, 1] > 0)[0]
            if len(lvo_ids) > 0: indices.append(lvo_ids[0].item())
            
            # 2. Tìm Lesion
            les_ids = torch.where(y_sums[:, 0] > 0)[0]
            for i in les_ids:
                if i.item() not in indices:
                    indices.append(i.item())
                    break
            
            # 3. Tìm CoW
            cow_ids = torch.where(y_sums[:, 2] > 0)[0]
            for i in cow_ids:
                if i.item() not in indices:
                    indices.append(i.item())
                    break
            
            # 4. Tìm Empty (No labels)
            empty_ids = torch.where(y_sums.sum(dim=1) == 0)[0]
            if len(empty_ids) > 0: indices.append(empty_ids[0].item())
            
            # Fill remaining to ensure 4 samples
            for i in range(x.shape[0]):
                if len(indices) >= 4: break
                if i not in indices:
                    indices.append(i)
            
            indices = indices[:4]

        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # Labels: 0=Lesion (Green), 1=LVO (Red), 2=CoW (Blue)
        colors = [
            (0, 1, 0), # Green for Lesion
            (1, 0, 0), # Red for LVO
            (0, 0, 1)  # Blue for CoW
        ]

        for row, idx in enumerate(indices):
            # 1. NCCT base with Brain Windowing (W:80, L:40)
            # Normalizing to [0, 1] range for visualization
            ncct = x_np[idx, 1]
            ncct = (ncct - ncct.min()) / (ncct.max() - ncct.min() + 1e-6)
            
            # 2. GT Overlay
            gt_overlay = np.stack([ncct]*3, axis=-1)
            for i in range(3):
                mask = y_np[idx, i] > 0.5
                gt_overlay[mask] = colors[i]
            
            # 3. Pred Overlay
            pred_overlay = np.stack([ncct]*3, axis=-1)
            for i in range(3):
                mask = p_np[i][idx, 0] > 0.5
                pred_overlay[mask] = colors[i]

            axes[row, 0].imshow(ncct, cmap="gray")
            axes[row, 0].set_title(f"NCCT (Sample {idx})")
            axes[row, 1].imshow(gt_overlay)
            axes[row, 1].set_title("Ground Truth (L-LVO-C)")
            axes[row, 2].imshow(pred_overlay)
            axes[row, 2].set_title("Prediction")
            
            for ax in axes[row]:
                ax.axis("off")

        save_path = self.ckpt_dir / "visualizations" / f"epoch_{epoch:02d}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"  --> Saved visualization to {save_path}")
