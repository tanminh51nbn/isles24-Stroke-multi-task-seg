"""
trainer.py — Full training pipeline for ISLES'24 Multi-Task Segmentation.

Architecture: HuggingFace Accelerate DDP (2×T4 GPU)
  - Each GPU runs its own process with its own model copy
  - Forward/backward run in PARALLEL on both GPUs
  - Only gradients are synchronized via AllReduce (NCCL Ring)
  - Result: ~1.85-1.95× speedup with 2 GPUs (vs ~1.2× with DataParallel)

Features:
  - Automatic Mixed Precision (managed by Accelerate, no manual GradScaler)
  - Composite Score: 0.4*Dice_Lesion + 0.4*Recall_LVO + 0.2*Dice_CoW
  - 3 Checkpoints: best_overall, best_lesion, best_lvo
  - Early stopping on Composite Score
  - Optional W&B logging with prediction visualization
"""
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

try:
    from accelerate import Accelerator
except ImportError:
    raise ImportError(
        "HuggingFace Accelerate is required. "
        "Install with: pip install accelerate"
    )

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Metric helpers
# ═══════════════════════════════════════════════════════════════

def _dice_score(logits: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Dice numerator and denominator for a batch.
    Returns (numerator, denominator) tensors for later reduction across GPUs.

    Dice = (2 * intersection) / (pred_sum + target_sum)

    Args:
        logits: [B, 1, H, W] raw logits
        target: [B, 1, H, W] binary {0, 1}

    Returns:
        inter:  scalar tensor — sum of intersections across batch
        union:  scalar tensor — sum of (pred + target) across batch
    """
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return inter, union


def _recall_score(logits: torch.Tensor, target: torch.Tensor,
                  threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Recall numerator and denominator for a batch.
    Returns (TP, TP+FN) tensors for later reduction across GPUs.

    Recall = TP / (TP + FN) = intersection / target_sum

    For LVO: "did the model touch/detect the clot?" — doesn't need
    precise boundaries, just needs to overlap with ground truth.

    Args:
        logits: [B, 1, H, W] raw logits
        target: [B, 1, H, W] binary {0, 1}

    Returns:
        tp:     scalar tensor — true positives (intersection)
        tp_fn:  scalar tensor — total positive pixels in target
    """
    pred = (torch.sigmoid(logits) > threshold).float()
    tp = (pred * target).sum()
    tp_fn = target.sum()
    return tp, tp_fn


# ═══════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════

def _create_overlay(ncct_slice: np.ndarray,
                    mask_true: np.ndarray,
                    mask_pred: np.ndarray) -> np.ndarray:
    """
    Create RGB overlay: NCCT (gray) + ground truth (green) + prediction (red).

    Args:
        ncct_slice: [H, W] normalized NCCT slice
        mask_true:  [H, W] binary ground truth
        mask_pred:  [H, W] binary prediction

    Returns:
        overlay: [H, W, 3] uint8 RGB image
    """
    # Normalize NCCT to [0, 255]
    ncct_norm = ncct_slice - ncct_slice.min()
    if ncct_norm.max() > 0:
        ncct_norm = ncct_norm / ncct_norm.max()
    gray = (ncct_norm * 255).astype(np.uint8)

    # Create RGB
    overlay = np.stack([gray, gray, gray], axis=-1)

    # Green = ground truth, Red = prediction
    alpha = 0.4
    gt_mask = mask_true > 0.5
    pred_mask = mask_pred > 0.5

    overlay[gt_mask, 1] = np.clip(
        overlay[gt_mask, 1] * (1 - alpha) + 255 * alpha, 0, 255
    ).astype(np.uint8)
    overlay[pred_mask, 0] = np.clip(
        overlay[pred_mask, 0] * (1 - alpha) + 255 * alpha, 0, 255
    ).astype(np.uint8)

    return overlay


# ═══════════════════════════════════════════════════════════════
#  Trainer
# ═══════════════════════════════════════════════════════════════

class MultiTaskTrainer:
    """
    Full training pipeline for ISLES'24 Multi-Task Segmentation.

    Uses HuggingFace Accelerate for:
      - DDP (Distributed Data Parallel) across 2×T4 GPUs
      - Automatic Mixed Precision (no manual GradScaler needed)
      - Gradient synchronization via AllReduce

    Must be instantiated INSIDE the function passed to notebook_launcher.
    """

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
        """
        Args:
            model:        MultiTaskSharedUNet instance
            criterion:    MultiTaskLoss instance
            optimizer:    AdamW with differential LR
            scheduler:    CosineWarmup scheduler
            train_loader: Training DataLoader
            val_loader:   Validation DataLoader
            cfg:          Full training config dict (from train.yaml)
        """
        train_cfg = cfg["training"]

        # ── Create Accelerator (handles DDP + AMP) ──
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        
        self.accelerator = Accelerator(
            mixed_precision="fp16" if train_cfg.get("amp", True) else "no",
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
            kwargs_handlers=[ddp_kwargs]
        )

        # ── Prepare all components for distributed training ──
        # Accelerate wraps model in DDP, splits data across GPUs,
        # and injects AMP autocast + GradScaler automatically.
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

        # Criterion has no parameters → no prepare needed
        self.criterion = criterion

        # ── Config ──
        self.epochs = train_cfg["epochs"]
        self.grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)

        # Composite score weights
        comp_cfg = train_cfg.get("composite_weights", {})
        self.comp_w_dice_lesion = comp_cfg.get("dice_lesion", 0.4)
        self.comp_w_recall_lvo = comp_cfg.get("recall_lvo", 0.4)
        self.comp_w_dice_cow = comp_cfg.get("dice_cow", 0.2)

        # Checkpointing
        ckpt_cfg = train_cfg.get("checkpoint", {})
        self.ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints"))
        self.save_best_overall = ckpt_cfg.get("save_best_overall", True)
        self.save_best_lesion = ckpt_cfg.get("save_best_lesion", True)
        self.save_best_lvo = ckpt_cfg.get("save_best_lvo", True)

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.es_enabled = es_cfg.get("enabled", True)
        self.es_patience = es_cfg.get("patience", 15)
        self.es_min_delta = es_cfg.get("min_delta", 0.001)

        # Logging
        log_cfg = train_cfg.get("logging", {})
        self.log_interval = log_cfg.get("log_interval", 10)
        self.viz_interval = log_cfg.get("visualize_every", 5)

        # W&B
        wandb_cfg = log_cfg.get("wandb", {})
        self.wandb_enabled = (
            wandb_cfg.get("enabled", False)
            and WANDB_AVAILABLE
            and self.accelerator.is_main_process
        )

        # ── State tracking ──
        self.best_composite = 0.0
        self.best_dice_lesion = 0.0
        self.best_recall_lvo = 0.0
        self.patience_counter = 0

        # ── Init ──
        if self.accelerator.is_main_process:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.wandb_enabled:
            wandb.init(
                project=wandb_cfg.get("project", "isles24-stroke"),
                name=wandb_cfg.get("run_name"),
                config=cfg,
            )

        self.accelerator.print(
            f"Trainer initialized: {self.epochs} epochs, "
            f"{self.accelerator.num_processes} GPUs, "
            f"AMP={'fp16' if train_cfg.get('amp') else 'off'}"
        )

    # ─────────────────────────────────────────────────────────────
    #  Main training loop
    # ─────────────────────────────────────────────────────────────

    def train(self) -> float:
        """
        Run full training: train → validate → checkpoint → early stop.

        Returns:
            Best composite score achieved
        """
        self.accelerator.print("=" * 60)
        self.accelerator.print("  ISLES'24 Multi-Task Training")
        self.accelerator.print("=" * 60)

        for epoch in range(self.epochs):
            t0 = time.time()

            # ── Train ──
            train_loss, train_task_losses = self._train_one_epoch(epoch)

            # ── Validate ──
            val_loss, val_metrics = self._validate(epoch)

            # ── Scheduler step (per epoch) ──
            self.scheduler.step()

            # ── Compute composite score ──
            composite = self._compute_composite(val_metrics)
            val_metrics["composite_score"] = composite

            elapsed = time.time() - t0

            # ── Log ──
            self._log_epoch(
                epoch, train_loss, train_task_losses,
                val_loss, val_metrics, elapsed,
            )

            # ── Checkpoint (3 independent best trackers) ──
            self._checkpoint(epoch, val_metrics)

            # ── Early stopping ──
            if self._check_early_stopping(composite):
                self.accelerator.print(
                    f"\n⚡ Early stopping at epoch {epoch} "
                    f"(no improvement for {self.es_patience} epochs)"
                )
                break

        # ── Cleanup ──
        if self.wandb_enabled:
            wandb.finish()

        self.accelerator.print(
            f"\nTraining complete! Best composite: {self.best_composite:.4f}"
        )
        return self.best_composite

    # ─────────────────────────────────────────────────────────────
    #  Train one epoch
    # ─────────────────────────────────────────────────────────────

    def _train_one_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.

        Returns:
            avg_loss:    mean total loss over all batches
            task_losses: {"lesion": float, "lvo": float, "cow": float}
        """
        self.model.train()

        running_loss = 0.0
        task_loss_sums = {t: 0.0 for t in self.TASK_NAMES}
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:02d} [Train]",
            disable=not self.accelerator.is_main_process,
        )

        for step, (x, y, brain_mask) in enumerate(pbar):
            with self.accelerator.accumulate(self.model):
                # Forward (AMP autocast managed by Accelerate)
                with self.accelerator.autocast():
                    preds = self.model(x)
                    
                    # ── Apply Brain Mask to Logits ──
                    brain_mask = brain_mask.to(preds[0].device)
                    for i in range(len(preds)):
                        preds[i] = preds[i] * brain_mask + (-1e9) * (1 - brain_mask)
                        
                    loss, loss_dict = self.criterion(preds, y)

                # Backward (Accelerate handles gradient scaling)
                self.accelerator.backward(loss)

                # Gradient clipping (prevents LVO gradient explosion)
                if self.grad_clip_norm > 0 and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )

                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Accumulate
            running_loss += loss.item()
            for t in self.TASK_NAMES:
                task_loss_sums[t] += loss_dict[t]
            n_batches += 1

            # Progress bar
            if step % self.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{self.optimizer.param_groups[-1]['lr']:.2e}",
                )

        avg_loss = running_loss / max(n_batches, 1)
        avg_tasks = {t: v / max(n_batches, 1) for t, v in task_loss_sums.items()}
        return avg_loss, avg_tasks

    # ─────────────────────────────────────────────────────────────
    #  Validate
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Validate and compute per-task metrics.

        Metrics computed:
          - Dice Score:   Lesion, CoW
          - Recall:       LVO (clinically: detect the clot, don't need perfect mask)
          - Composite:    0.4*Dice_Lesion + 0.4*Recall_LVO + 0.2*Dice_CoW

        Metrics are reduced across GPUs via AllReduce before final computation.

        Returns:
            avg_loss: mean validation loss
            metrics:  {"dice_lesion", "recall_lvo", "dice_cow"} — all in [0, 1]
        """
        self.model.eval()
        device = self.accelerator.device

        running_loss = 0.0
        n_batches = 0

        # Accumulators for reduction: [inter_lesion, union_lesion,
        #                              tp_lvo, tp_fn_lvo,
        #                              inter_cow, union_cow]
        accum = torch.zeros(6, device=device)

        # Store one batch for visualization
        viz_data = None

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch:02d} [Val]  ",
            disable=not self.accelerator.is_main_process,
        )

        for x, y, brain_mask in pbar:
            with self.accelerator.autocast():
                preds = self.model(x)
                
                # ── Apply Brain Mask to Logits ──
                brain_mask = brain_mask.to(preds[0].device)
                for i in range(len(preds)):
                    preds[i] = preds[i] * brain_mask + (-1e9) * (1 - brain_mask)
                    
                loss, _ = self.criterion(preds, y)

            running_loss += loss.item()
            n_batches += 1

            # Dice for lesion (task 0)
            inter, union = _dice_score(preds[0], y[:, 0:1])
            accum[0] += inter
            accum[1] += union

            # Recall for LVO (task 1)
            tp, tp_fn = _recall_score(preds[1], y[:, 1:2])
            accum[2] += tp
            accum[3] += tp_fn

            # Dice for CoW (task 2)
            inter, union = _dice_score(preds[2], y[:, 2:3])
            accum[4] += inter
            accum[5] += union

            # Save first batch for visualization
            if viz_data is None:
                viz_data = (
                    x[:1].detach(),
                    y[:1].detach(),
                    [p[:1].detach() for p in preds],
                )

        # ── Reduce across GPUs ──
        accum = self.accelerator.reduce(accum, reduction="sum")

        smooth = 1e-6
        dice_lesion = (2.0 * accum[0] + smooth) / (accum[1] + smooth)
        recall_lvo = (accum[2] + smooth) / (accum[3] + smooth)
        dice_cow = (2.0 * accum[4] + smooth) / (accum[5] + smooth)

        metrics = {
            "dice_lesion": dice_lesion.item(),
            "recall_lvo": recall_lvo.item(),
            "dice_cow": dice_cow.item(),
        }

        # ── Visualization ──
        if viz_data is not None and epoch % self.viz_interval == 0:
            self._visualize(epoch, *viz_data)

        avg_loss = running_loss / max(n_batches, 1)
        return avg_loss, metrics

    # ─────────────────────────────────────────────────────────────
    #  Composite Score
    # ─────────────────────────────────────────────────────────────

    def _compute_composite(self, metrics: Dict[str, float]) -> float:
        """
        Composite = 0.4 × Dice_Lesion + 0.4 × Recall_LVO + 0.2 × Dice_CoW

        Clinically meaningful weighting:
          - Lesion Dice:  how accurately we measure infarct volume
          - LVO Recall:   did we detect the clot? (critical for emergency)
          - CoW Dice:     collateral vessel mapping (supplementary)
        """
        return (
            self.comp_w_dice_lesion * metrics.get("dice_lesion", 0)
            + self.comp_w_recall_lvo * metrics.get("recall_lvo", 0)
            + self.comp_w_dice_cow * metrics.get("dice_cow", 0)
        )

    # ─────────────────────────────────────────────────────────────
    #  Checkpointing
    # ─────────────────────────────────────────────────────────────

    def _checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """
        Save up to 3 best models independently.

        Each checkpoint tracks its own "best" independently:
          - best_overall.pth  → highest Composite Score (main release)
          - best_lesion.pth   → highest Dice Lesion (volume measurement)
          - best_lvo.pth      → highest Recall LVO (emergency detection)
        """
        if not self.accelerator.is_main_process:
            return

        unwrapped = self.accelerator.unwrap_model(self.model)

        def _save(filename: str, monitor_name: str, monitor_value: float):
            path = self.ckpt_dir / filename
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unwrapped.state_dict(),
                    "monitor": monitor_name,
                    "monitor_value": monitor_value,
                    "metrics": metrics,
                },
                path,
            )
            logger.info(f"  💾 Saved {filename} ({monitor_name}={monitor_value:.4f})")

        composite = metrics.get("composite_score", 0)
        dice_lesion = metrics.get("dice_lesion", 0)
        recall_lvo = metrics.get("recall_lvo", 0)

        # 1. Best overall (Composite Score)
        if self.save_best_overall and composite > self.best_composite:
            _save("best_overall.pth", "composite_score", composite)
            self.best_composite = composite

        # 2. Best lesion (Dice Lesion)
        if self.save_best_lesion and dice_lesion > self.best_dice_lesion:
            _save("best_lesion.pth", "dice_lesion", dice_lesion)
            self.best_dice_lesion = dice_lesion

        # 3. Best LVO (Recall LVO)
        if self.save_best_lvo and recall_lvo > self.best_recall_lvo:
            _save("best_lvo.pth", "recall_lvo", recall_lvo)
            self.best_recall_lvo = recall_lvo

    # ─────────────────────────────────────────────────────────────
    #  Early Stopping
    # ─────────────────────────────────────────────────────────────

    def _check_early_stopping(self, composite: float) -> bool:
        """
        Check if training should stop based on Composite Score plateau.

        Returns True if patience exceeded (should stop).
        """
        if not self.es_enabled:
            return False

        if composite > (self.best_composite - self.es_min_delta):
            # Composite improved (or very close) → checkpoint already updated
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.es_patience

    # ─────────────────────────────────────────────────────────────
    #  Logging
    # ─────────────────────────────────────────────────────────────

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_tasks: Dict[str, float],
        val_loss: float,
        val_metrics: Dict[str, float],
        elapsed: float,
    ):
        """Log epoch summary to console and optionally W&B."""
        if not self.accelerator.is_main_process:
            return

        composite = val_metrics.get("composite_score", 0)
        lr = self.optimizer.param_groups[-1]["lr"]

        # Console
        self.accelerator.print(
            f"Epoch {epoch:02d} │ "
            f"train_loss={train_loss:.4f} │ "
            f"val_loss={val_loss:.4f} │ "
            f"dice_les={val_metrics.get('dice_lesion', 0):.4f} │ "
            f"rec_lvo={val_metrics.get('recall_lvo', 0):.4f} │ "
            f"dice_cow={val_metrics.get('dice_cow', 0):.4f} │ "
            f"composite={composite:.4f} │ "
            f"lr={lr:.2e} │ "
            f"{elapsed:.0f}s │ "
            f"patience={self.patience_counter}/{self.es_patience}"
        )

        # W&B
        if self.wandb_enabled:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/loss_lesion": train_tasks.get("lesion", 0),
                "train/loss_lvo": train_tasks.get("lvo", 0),
                "train/loss_cow": train_tasks.get("cow", 0),
                "val/loss": val_loss,
                "val/dice_lesion": val_metrics.get("dice_lesion", 0),
                "val/recall_lvo": val_metrics.get("recall_lvo", 0),
                "val/dice_cow": val_metrics.get("dice_cow", 0),
                "val/composite_score": composite,
                "lr": lr,
                "epoch_time_s": elapsed,
            }
            wandb.log(log_dict, step=epoch)

    # ─────────────────────────────────────────────────────────────
    #  Visualization
    # ─────────────────────────────────────────────────────────────

    def _visualize(
        self,
        epoch: int,
        x: torch.Tensor,
        y_true: torch.Tensor,
        preds: list,
    ):
        """
        Overlay predicted masks on NCCT slice.

        Green = ground truth, Red = prediction.
        Helps detect "background collapse" (model predicting all zeros).

        Only runs on main process.
        """
        if not self.accelerator.is_main_process:
            return

        try:
            # NCCT middle slice (channel 1 of 0-2 in input)
            ncct = x[0, 1].float().cpu().numpy()

            for i, task_name in enumerate(self.TASK_NAMES):
                mask_true = y_true[0, i].float().cpu().numpy()
                mask_pred = (
                    (torch.sigmoid(preds[i][0, 0]) > 0.5).float().cpu().numpy()
                )

                overlay = _create_overlay(ncct, mask_true, mask_pred)

                if self.wandb_enabled:
                    wandb.log(
                        {f"viz/{task_name}": wandb.Image(overlay)},
                        step=epoch,
                    )
                else:
                    # Save as numpy file (lightweight, no PIL dependency)
                    viz_dir = self.ckpt_dir / "viz"
                    viz_dir.mkdir(exist_ok=True)
                    np.save(
                        viz_dir / f"epoch{epoch:02d}_{task_name}.npy",
                        overlay,
                    )

        except Exception as e:
            logger.warning(f"Visualization failed at epoch {epoch}: {e}")
