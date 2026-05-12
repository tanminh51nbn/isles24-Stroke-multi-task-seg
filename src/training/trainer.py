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

from compile.metrics import compute_all_metrics, finalize_lvo_f1, accumulate_patient_lvo_stats, finalize_patient_lvo_acc
from evaluation.visualize import overlay_predictions, select_best_sample
from data.fold_split import apply_sampling
import os

class Trainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, train_files_original, loss_fn, optimizer, scheduler, config: dict, device: torch.device, rank: int = 0):
        self.model, self.train_loader, self.val_loader = model, train_loader, val_loader
        self.train_files_original = train_files_original
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
        max_lvo_spike = 0.0  # [FIX 1.3] Track spike lớn nhất, log 1 lần / epoch
        print(f"\nStarting Epoch {epoch + 1}:")
        
        for batch_idx, batch in enumerate(self.train_loader):
            inp = batch["input"].to(self.device, non_blocking=True)
            lbl = batch["label"].to(self.device, non_blocking=True)
            if not torch.isfinite(inp).all(): inp = torch.nan_to_num(inp, nan=0.0)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                preds = self.model(inp, epoch=epoch)
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

            # [FIX 1.3] Per-task gradient clip cho LVO HEAD trước global clip.
            # LVO gradient spike 200-986 trong log trước gây LVO head "reset" định kỳ.
            # Clip LVO params riêng → rồi mới global clip toàn model.
            raw = self.model.module if hasattr(self.model, "module") else self.model
            lvo_params = [p for n, p in raw.named_parameters()
                          if "lvo" in n.lower() and p.grad is not None]
            if lvo_params:
                lvo_norm = nn.utils.clip_grad_norm_(lvo_params, max_norm=10.0)
                lvo_val = lvo_norm.item() if torch.is_tensor(lvo_norm) else float(lvo_norm)
                if lvo_val > max_lvo_spike:
                    max_lvo_spike = lvo_val
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["total"].item()
            n_batches += 1

        # [FIX 1.3] Log spike LVO 1 lần / epoch (chỉ khi > 100 — spike thực sự nguy hiểm)
        if self.rank == 0 and max_lvo_spike > 100.0:
            import math
            spike_str = "inf" if math.isinf(max_lvo_spike) else f"{max_lvo_spike:.1f}"
            print(f"    [WARN] LVO max grad spike: {spike_str} → clipped to 10.0")
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
        sum_d_l, sum_d_c = 0.0, 0.0
        sum_aad, sum_alcd = 0.0, 0.0
        sum_p_v = 0.0
        n_b = 0
        
        # [FIX] Global LVO stats: gom TP/FP/FN trên toàn Val set thay vì per-batch avg
        lvo_stats = {"tp": 0, "fp": 0, "fn": 0}
        # Patient-level LVO stats (chỉ để log, không ảnh hưởng training)
        patient_stats = {}
        
        vis_interval = self.config["training"]["logging"].get("visualize_every", 5)
        should_vis = (epoch % vis_interval == 0) and (self.rank == 0)
        vis_candidates = []  # Thu thập ứng viên từ toàn bộ val loop

        for batch in self.val_loader:
            inp = batch["input"].to(self.device, non_blocking=True)
            lbl = batch["label"].to(self.device, non_blocking=True)
            if not torch.isfinite(inp).all(): continue

            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                preds = self.model(inp, epoch=epoch)
                losses = self.loss_fn(preds, lbl, epoch=epoch, batch_idx=-1)

            if not torch.isfinite(losses["total"]): continue

            total_loss += losses["total"].item()
            sum_p_v += losses.get("p_lvo", 1.0)
            
            # Truyền lvo_stats vào để gom TP/FP/FN toàn cục
            metrics = compute_all_metrics(preds, lbl, self.metric_weights, lvo_stats=lvo_stats)
            sum_d_l  += metrics["dice_lesion"]
            sum_d_c  += metrics["dice_cow"]
            sum_aad  += metrics["aad_lesion"]
            sum_alcd += metrics["alcd_lesion"]
            n_b += 1

            # Gom patient-level stats (chỉ rank 0)
            if self.rank == 0:
                paths = batch.get("path", [""] * inp.shape[0])
                accumulate_patient_lvo_stats(
                    preds["lvo"], lbl[:, 1:2], paths, patient_stats,
                    threshold=self.metric_weights.get("thresholds", {}).get("lvo", 0.2)
                )

            # Thu thập ứng viên visualize (chỉ rank 0, từ tất cả batch của val loop)
            if should_vis and self.rank == 0:
                for i in range(inp.shape[0]):
                    vis_candidates.append({
                        "input": inp[i].cpu(),
                        "label": lbl[i].cpu(),
                        "pred":  {k: v[i:i+1].cpu() for k, v in preds.items() if isinstance(v, torch.Tensor)},
                        "path":  batch.get("path", [""] * inp.shape[0])[i],
                    })

        # [FIX] Đồng bộ TP/FP/FN qua DDP trước khi tính F1
        if dist.is_initialized():
            lvo_tensor = torch.tensor(
                [lvo_stats["tp"], lvo_stats["fp"], lvo_stats["fn"]], 
                dtype=torch.float32, device=self.device
            )
            dist.all_reduce(lvo_tensor, op=dist.ReduceOp.SUM)
            lvo_stats = {"tp": int(lvo_tensor[0].item()), "fp": int(lvo_tensor[1].item()), "fn": int(lvo_tensor[2].item())}
            
            sync = torch.tensor([total_loss, sum_d_l, sum_d_c, sum_aad, sum_alcd, sum_p_v, float(n_b)], device=self.device)
            dist.all_reduce(sync, op=dist.ReduceOp.SUM)
            v = sync.cpu().numpy()
            avg_l, ad_l, ad_c, a_aad, a_alcd, ap_v = v[0]/max(v[6],1), v[1]/max(v[6],1), v[2]/max(v[6],1), v[3]/max(v[6],1), v[4]/max(v[6],1), v[5]/max(v[6],1)
        else:
            avg_l = total_loss/max(n_b,1)
            ad_l  = sum_d_l/max(n_b,1)
            ad_c  = sum_d_c/max(n_b,1)
            a_aad = sum_aad/max(n_b,1)
            a_alcd = sum_alcd/max(n_b,1)
            ap_v  = sum_p_v/max(n_b,1)

        # Tính F1 Global một lần duy nhất sau khi đã gom toàn bộ Val set
        af1_v = finalize_lvo_f1(lvo_stats)
        if self.rank == 0:
            print(f"    [LVO_GLOBAL] TP={lvo_stats['tp']} FP={lvo_stats['fp']} FN={lvo_stats['fn']} => F1={af1_v:.2f}%")
            # Log patient-level accuracy
            pat = finalize_patient_lvo_acc(
                patient_stats,
                threshold=self.metric_weights.get("thresholds", {}).get("lvo", 0.2)
            )
            print(f"    [LVO_PATIENT] Đúng: {pat['tp']+pat['tn']}/{pat['n']} bệnh nhân "
                  f"({pat['accuracy']*100:.1f}%)  TP={pat['tp']} FP={pat['fp']} FN={pat['fn']} TN={pat['tn']}")
            # Visualize sample tốt nhất (sau khi đã dưắt toàn bộ val loop)
            if should_vis and vis_candidates:
                best = select_best_sample(vis_candidates)
                if best is not None:
                    vis_dir = os.path.join(self.output_dir, self.config["training"]["checkpoint"]["dir"], "visualizations")
                    os.makedirs(vis_dir, exist_ok=True)
                    overlay_predictions(
                        sample={"input": best["input"], "label": best["label"], "path": best["path"]},
                        preds=best["pred"],
                        epoch=epoch, save_dir=vis_dir, thresholds=self.metric_weights.get("thresholds", {})
                    )

        w = self.metric_weights
        comp = (w["dice_lesion_weight"] * ad_l + w["f1_lvo_weight"] * (af1_v/100.0) + w["dice_cow_weight"] * ad_c)
        
        return {
            "val_loss": avg_l, "dice_lesion": ad_l, "f1_lvo": af1_v, "dice_cow": ad_c, 
            "aad_lesion": a_aad, "alcd_lesion": a_alcd,
            "composite": comp, 
            "p_lesion": float(losses.get("p_lesion", 1.0)), 
            "p_lvo": ap_v, 
            "p_cow": float(losses.get("p_cow", 1.0))
        }

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
            # [CYCLIC STRIDE] Cập nhật danh sách file huấn luyện cho Epoch mới
            new_train_list = apply_sampling(self.train_files_original, self.config, epoch=epoch)
            self.train_loader.dataset.file_list = new_train_list
            
            if epoch == self.freeze_enc_epochs: raw.unfreeze_encoders()
            if hasattr(self.train_loader.sampler, "set_epoch"): self.train_loader.sampler.set_epoch(epoch)
            t_m = self.train_one_epoch(epoch)
            v_m = self.validate(epoch + 1)
            if self.rank == 0:
                lr_enc = self.optimizer.param_groups[0]['lr']
                lr_dec = self.optimizer.param_groups[1]['lr']
                print(f"{'-'*80}\n=> | [Ep {epoch+1:03d}/{self.epochs}] | LR (En/De): {lr_enc:.1e}/{lr_dec:.1e} | Comp: {v_m['composite']:.4f}")
                print(f"   | [VAL] Dice_Lesion: {v_m['dice_lesion']:.4f} | F1_LVO: {v_m['f1_lvo']/100.0:.4f} | Dice_CoW: {v_m['dice_cow']:.4f}")
                print(f"   | [VAL] Loss: {v_m['val_loss']:.4f} | AAD: {v_m['aad_lesion']:.2f}% | ALCD: {v_m['alcd_lesion']:.4f}")
                print(f"   | [TRA] Loss: {t_m['train_loss']:.4f} | P_LVO: {v_m['p_lvo']:.2f}\n{'-'*80}", flush=True)
            self.history.append({**t_m, **v_m, "epoch": epoch + 1})
            if checkpoint and self.rank == 0:
                if (epoch + 1) >= self.config["training"]["checkpoint"].get("start_epoch", 1):
                    checkpoint.update(self.model, self.optimizer, epoch + 1, v_m, scheduler=self.scheduler, history=self.history)
            if early_stopping and (epoch + 1) >= self.config["training"]["early_stopping"].get("start_epoch", 1):
                if early_stopping(v_m["composite"]): break
            self.scheduler.step()
            if hasattr(self.loss_fn, "update_weights_from_metrics") and self.rank == 0:
                self.loss_fn.update_weights_from_metrics(v_m, epoch)
        return self.history
