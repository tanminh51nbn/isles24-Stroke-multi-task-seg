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
def _dice_score(logits, target, threshold=0.5):
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return inter, union

def _recall_score(logits, target, threshold=0.5):
    pred = (torch.sigmoid(logits) > threshold).float()
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
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

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

            train_loss, train_tasks, grad_norm = self._train_one_epoch(epoch)
            val_loss, val_metrics = self._validate(epoch)

            # Visualization every N epochs
            visualize_every = self.cfg["training"].get("logging", {}).get("visualize_every", 5)
            if (epoch + 1) % visualize_every == 0 and self.accelerator.is_main_process:
                self._visualize_results(epoch)

            elapsed = time.time() - start_t
            self.scheduler.step()

            # Logging & Checkpointing (Main Process)
            if self.accelerator.is_main_process:
                composite = val_metrics.get("composite_score", 0)
                lr = self.optimizer.param_groups[-1]["lr"]
                
                logger.info(
                    f"Epoch {epoch:02d} | "
                    f"loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"dice_les={val_metrics.get('dice_lesion', 0):.4f} | "
                    f"rec_lvo={val_metrics.get('recall_lvo', 0):.4f} | "
                    f"grad={grad_norm:.2f} | "
                    f"lr={lr:.2e} | "
                    f"{elapsed:.0f}s"
                )

                # Checkpoint
                if composite > self.best_composite:
                    self.best_composite = composite
                    self.patience_counter = 0
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                        "composite": composite
                    }, self.ckpt_dir / "best_model.pth")
                    logger.info(f"  --> Saved new best model (Composite: {composite:.4f})")
                else:
                    self.patience_counter += 1

                # Early Stopping
                if self.patience_counter >= self.es_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
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

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]", disable=not self.accelerator.is_local_main_process)

        for step, (x, y, brain_mask) in enumerate(pbar):
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
                    if isinstance(current_grad_norm, torch.Tensor):
                        current_grad_norm = current_grad_norm.item()
                    total_grad_norm += current_grad_norm
                    grad_steps += 1

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            real_loss = loss.item()
            running_loss += real_loss
            n_batches += 1
            for t in self.TASK_NAMES:
                task_loss_sums[t] += loss_dict[t]

            if step % self.log_interval == 0:
                pbar.set_postfix(loss=f"{real_loss:.4f}")

        avg_loss = running_loss / max(n_batches, 1)
        avg_grad = total_grad_norm / max(grad_steps, 1)
        avg_tasks = {t: v / max(n_batches, 1) for t, v in task_loss_sums.items()}
        
        # Sync loss across processes for consistent logging
        avg_loss_t = torch.tensor([avg_loss], device=self.accelerator.device)
        avg_loss = self.accelerator.reduce(avg_loss_t, reduction="mean").item()
        
        return avg_loss, avg_tasks, avg_grad

    @torch.no_grad()
    def _validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        running_loss = 0.0
        n_batches = 0
        accum = torch.zeros(6, device=self.accelerator.device)

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:02d} [Val]  ", disable=not self.accelerator.is_local_main_process)

        for x, y, brain_mask in pbar:
            with self.accelerator.autocast():
                preds = self.model(x)
                preds = list(preds)
                for i in range(len(preds)):
                    preds[i] = preds[i] * brain_mask + (-1e9) * (1 - brain_mask)
                loss, _ = self.criterion(preds, y)

            running_loss += loss.item()
            n_batches += 1
            
            # Dice calculations (on GPU)
            dice_les = _dice_score(preds[0], y[:, 0:1], brain_mask)
            rec_lvo  = _recall_score(preds[1], y[:, 1:2], brain_mask)
            dice_cow = _dice_score(preds[2], y[:, 2:3], brain_mask)
            
            accum[0] += dice_les
            accum[1] += rec_lvo
            accum[2] += dice_cow
            accum[5] += 1

        # Global sync
        accum = self.accelerator.reduce(accum, reduction="sum")
        avg_loss_t = torch.tensor([running_loss / max(n_batches, 1)], device=self.accelerator.device)
        avg_loss = self.accelerator.reduce(avg_loss_t, reduction="mean").item()

        metrics = {}
        if self.accelerator.is_main_process:
            accum = accum.cpu().numpy()
            dice_les = 2.0 * accum[0] / max(accum[1], 1e-6)
            rec_lvo = accum[2] / max(accum[3], 1e-6)
            dice_cow = 2.0 * accum[4] / max(accum[5], 1e-6)
            
            comp = 0.4 * dice_les + 0.4 * rec_lvo + 0.2 * dice_cow
            
            metrics = {
                "dice_lesion": dice_les,
                "recall_lvo": rec_lvo,
                "dice_cow": dice_cow,
                "composite_score": comp
            }
            
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
        
        # Chọn 4 ảnh để hiển thị (ưu tiên những ảnh có LVO hoặc Lesion)
        indices = []
        for i in range(x_np.shape[0]):
            if y_np[i, 1].max() > 0 or y_np[i, 0].max() > 0:
                indices.append(i)
        
        if len(indices) < 4:
            indices.extend(list(range(min(4 - len(indices), x_np.shape[0]))))
        indices = indices[:4]

        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # Labels: 0=Lesion (Green), 1=LVO (Red), 2=CoW (Blue)
        colors = [ (0, 1, 0), (1, 0, 0), (0, 0, 1) ]

        for row, idx in enumerate(indices):
            # 1. NCCT base (Channel index 1 is central NCCT slice)
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
