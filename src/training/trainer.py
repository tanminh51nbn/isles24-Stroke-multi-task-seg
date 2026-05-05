"""
trainer.py — Vòng lặp huấn luyện chính cho Dual-Encoder Multi-Task UNet
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Optional
import math

from compile.metrics import compute_all_metrics
from evaluation.visualize import overlay_predictions

class Trainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, loss_fn, optimizer, scheduler, config: dict, device: torch.device, rank: int = 0):
        self.model, self.train_loader, self.val_loader = model, train_loader, val_loader
        self.loss_fn, self.optimizer, self.scheduler = loss_fn, optimizer, scheduler
        self.config, self.device, self.rank = config, device, rank
        self.output_dir = config.get("output_dir", "outputs")
        
        t_cfg = config["training"]
        self.epochs = int(t_cfg["epochs"])
        self.amp_enabled = bool(t_cfg["amp"])
        self.grad_clip_norm = float(t_cfg["grad_clip_norm"])
        self.freeze_enc_epochs = int(t_cfg["freeze_encoder_epochs"])
        self.log_interval = int(t_cfg["logging"]["log_interval"])
        self.metric_weights = config["composite_score"]

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled)
        self.history = []

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss, n_batches, nan_batches = 0.0, 0, 0
        print(f"\nStarting Epoch {epoch + 1}:")
        
        for batch_idx, batch in enumerate(self.train_loader):
            inp = batch["input"].to(self.device, non_blocking=True)
            lbl = batch["label"].to(self.device, non_blocking=True)
            if not torch.isfinite(inp).all(): inp = torch.nan_to_num(inp, nan=0.0)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                preds = self.model(inp)
                losses = self.loss_fn(preds, lbl, epoch=epoch, batch_idx=batch_idx)

            if not torch.isfinite(losses["total"]):
                nan_batches += 1
                continue

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            
            if batch_idx % self.log_interval == 0 and self.rank == 0:
                gn = {"l": 0.0, "v": 0.0, "c": 0.0}
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        val = p.grad.detach().norm(2).item()
                        if "lesion" in n.lower(): gn["l"] += val
                        elif "lvo" in n.lower(): gn["v"] += val
                        elif "cow" in n.lower(): gn["c"] += val
                print(f"    [GRAD_DEBUG] B{batch_idx} Norms: LVO={gn['v']:.4f}, Lesion={gn['l']:.4f}, CoW={gn['c']:.4f}")

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["total"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if dist.is_initialized():
            sync = torch.tensor([total_loss, float(n_batches)], device=self.device)
            dist.all_reduce(sync, op=dist.ReduceOp.SUM)
            avg_loss = sync[0].item() / max(sync[1].item(), 1)
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0.0
        sum_d_l, sum_r_v, sum_d_c = 0.0, 0.0, 0.0
        sum_p_v, sum_s_v = 0.0, 0.0
        n_b, n_v_b = 0, 0
        
        vis_interval = self.config["training"]["logging"].get("visualize_every", 5)
        should_vis = (epoch % vis_interval == 0) and (self.rank == 0)
        visualized = False

        for batch_idx, batch in enumerate(self.val_loader):
            inp = batch["input"].to(self.device, non_blocking=True)
            lbl = batch["label"].to(self.device, non_blocking=True)
            if not torch.isfinite(inp).all(): continue

            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                preds = self.model(inp)
                losses = self.loss_fn(preds, lbl, epoch=epoch)

            if not torch.isfinite(losses["total"]): continue

            total_loss += losses["total"].item()
            sum_p_v += losses.get("p_lvo", 1.0)
            sum_s_v += losses.get("sigma_lvo", 1.0)
            
            metrics = compute_all_metrics(preds, lbl, self.metric_weights)
            sum_d_l += metrics["dice_lesion"]
            sum_d_c += metrics["dice_cow"]
            n_b += 1
            if metrics["recall_lvo"] >= 0:
                sum_r_v += metrics["recall_lvo"]; n_v_b += 1

            if should_vis and not visualized:
                vis_dir = os.path.join(self.output_dir, self.config["training"]["checkpoint"]["dir"], "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                overlay_predictions(
                    sample={"input": inp[0], "label": lbl[0], "path": batch["path"][0]},
                    preds={k: v[0:1] for k, v in preds.items() if isinstance(v, torch.Tensor)},
                    epoch=epoch, save_dir=vis_dir, thresholds=self.metric_weights.get("thresholds", {})
                )
                visualized = True

        if dist.is_initialized():
            sync = torch.tensor([total_loss, sum_d_l, sum_r_v, sum_d_c, sum_p_v, sum_s_v, float(n_b), float(n_v_b)], device=self.device)
            dist.all_reduce(sync, op=dist.ReduceOp.SUM)
            v = sync.cpu().numpy()
            avg_l, ad_l, ar_v, ad_c, ap_v, as_v = v[0]/max(v[6],1), v[1]/max(v[6],1), v[2]/max(v[7],1), v[3]/max(v[6],1), v[4]/max(v[6],1), v[5]/max(v[6],1)
        else:
            avg_l, ad_l, ar_v, ad_c, ap_v, as_v = total_loss/max(n_b,1), sum_d_l/max(n_b,1), sum_r_v/max(n_v_b,1), sum_d_c/max(n_b,1), sum_p_v/max(n_b,1), sum_s_v/max(n_b,1)

        w = self.metric_weights
        comp = (w["w_lesion"] * ad_l + w["w_lvo"] * ar_v + w["w_cow"] * ad_c)
        return {"val_loss": avg_l, "dice_lesion": ad_l, "recall_lvo": ar_v, "dice_cow": ad_c, "composite": comp, "p_lvo": ap_v, "sigma_lvo": as_v}

    def load_checkpoint(self, path: str):
        if not os.path.exists(path): return 0
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        raw = self.model.module if hasattr(self.model, "module") else self.model
        raw.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and self.optimizer: self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and self.scheduler: self.scheduler.load_state_dict(ckpt["scheduler"])
        if "history" in ckpt: self.history = ckpt["history"]
        return ckpt.get("epoch", 0)

    def fit(self, early_stopping=None, checkpoint=None, start_epoch: int = 0):
        raw = self.model.module if hasattr(self.model, "module") else self.model
        raw.freeze_encoders()
        for epoch in range(start_epoch, self.epochs):
            if epoch == self.freeze_enc_epochs: raw.unfreeze_encoders()
            if hasattr(self.train_loader.sampler, "set_epoch"): self.train_loader.sampler.set_epoch(epoch)
            t_m = self.train_one_epoch(epoch)
            v_m = self.validate(epoch + 1)
            if self.rank == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"{'-'*80}\n=> | [Ep {epoch+1:03d}/{self.epochs}] | LR: {lr:.2e} | Comp: {v_m['composite']:.4f}")
                print(f"   | [VAL] L: {v_m['dice_lesion']:.4f} | V: {v_m['recall_lvo']:.4f} | C: {v_m['dice_cow']:.4f}")
                print(f"   | [TRN] Loss: {t_m['train_loss']:.4f} | P_V: {v_m['p_lvo']:.2f} | Sig_V: {v_m['sigma_lvo']:.2f}\n{'-'*80}", flush=True)
            self.history.append({**t_m, **v_m, "epoch": epoch + 1})
            if checkpoint and self.rank == 0:
                if (epoch + 1) >= self.config["training"]["checkpoint"].get("start_epoch", 1):
                    checkpoint.update(self.model, self.optimizer, epoch + 1, v_m, scheduler=self.scheduler, history=self.history)
            if early_stopping and (epoch + 1) >= self.config["training"]["early_stopping"].get("start_epoch", 1):
                if early_stopping(v_m["composite"]): break
            self.scheduler.step()
            if hasattr(self.loss_fn, "update_epoch_stats"): self.loss_fn.update_epoch_stats()
        return self.history
