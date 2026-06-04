"""
trainer.py — Vòng lặp huấn luyện chính cho Multi-Task UNet
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Optional
import math

from compile.metrics import (
    compute_all_metrics, finalize_lvo_dice, accumulate_lvo_stats,
    accumulate_patient_lvo_stats, finalize_patient_lvo_acc,
    get_lvo_threshold,
)
from evaluation.visualize import overlay_predictions, select_best_sample
from data.fold_split import apply_sampling


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
        from compile import PCGrad
        self.pcgrad = PCGrad(self.optimizer, use_amp=self.amp_enabled, max_norm=self.grad_clip_norm)
        # [DEBUG] Encoder param IDs cho phân tích gradient
        _raw = self.model.module if hasattr(self.model, "module") else self.model
        self._enc_param_ids = {id(p) for p in _raw.encoder.parameters()} if hasattr(_raw, 'encoder') else set()

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss, main_loss, raw_loss, n_batches, nan_batches = 0.0, 0.0, 0.0, 0, 0
        sum_t_les, sum_t_lvo, sum_t_cow = 0.0, 0.0, 0.0
        max_lvo_spike = 0.0  # [FIX 1.3] Track spike lớn nhất, log 1 lần / epoch
        print(f"\nStarting Epoch {epoch + 1}:")
        
        for batch_idx, batch in enumerate(self.train_loader):
            inp = batch["input"].to(self.device, non_blocking=True)
            lbl = batch["label"].to(self.device, non_blocking=True)
            if not torch.isfinite(inp).all(): inp = torch.nan_to_num(inp, nan=0.0)

            # DIAGNOSTIC TRACKING
            if not hasattr(self, "_diag_lvo_tp"):
                self._diag_lvo_tp = self._diag_lvo_fp = self._diag_lvo_fn = 0
                self._diag_lvo_max = 0.0

            self.optimizer.zero_grad(set_to_none=True)
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            with torch.amp.autocast('cuda', enabled=self.amp_enabled):
                preds = raw_model(inp, epoch=epoch, decoupled=True)
                losses = self.loss_fn(preds, lbl, epoch=epoch, batch_idx=batch_idx)
                
                # Accumulate diagnostics
                with torch.no_grad():
                    lvo_p = torch.sigmoid(preds["lvo"])
                    self._diag_lvo_max = max(self._diag_lvo_max, lvo_p.max().item())
                    l_thr = get_lvo_threshold(epoch, self.metric_weights)
                    max_r = float(self.config.get("loss", {}).get("lvo", {}).get("max_radius", 10.0))
                    stats = accumulate_lvo_stats(preds["lvo"], lbl[:, 1:2], threshold=l_thr, max_radius=max_r)
                    self._diag_lvo_tp += stats["tp"]
                    self._diag_lvo_fp += stats["fp"]
                    self._diag_lvo_fn += stats["fn"]
            task_losses = [losses["total_lesion"], losses["total_lvo"], losses["total_cow"]]
            is_finite = torch.tensor(1.0 if all(torch.isfinite(l) for l in task_losses) else 0.0, device=self.device)
            if dist.is_initialized():
                dist.all_reduce(is_finite, op=dist.ReduceOp.MIN)
            
            if is_finite.item() == 0.0:
                nan_batches += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # Phẫu thuật Gradient bằng PCGrad
            _log_enc = (batch_idx % self.log_interval == 0 and self.rank == 0)
            weights = [losses["p_lesion"], losses["p_lvo"], losses["p_cow"]]
            self.pcgrad.backward_encoder_bypass(task_losses, self.model, scaler=self.scaler,
                                                encoder_debug_ids=self._enc_param_ids if _log_enc else None,
                                                weights=weights, asymmetric=True)
            self.scaler.unscale_(self.optimizer)
            
            if batch_idx % self.log_interval == 0 and self.rank == 0:
                gn = {"l": 0.0, "v": 0.0, "c": 0.0, "e": 0.0}
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        val = p.grad.detach().norm(2).item()
                        sq_val = val * val
                        if "lesion" in n.lower(): gn["l"] += sq_val
                        elif "lvo" in n.lower(): gn["v"] += sq_val
                        elif "cow" in n.lower(): gn["c"] += sq_val
                        elif "encoder" in n.lower() or "features" in n.lower(): gn["e"] += sq_val
                
                print(f"    [GRAD] B{batch_idx:03d} | 🔴 Les: {gn['l']**0.5:.2f} 🎯 LVO: {gn['v']**0.5:.2f} 🟢 CoW: {gn['c']**0.5:.2f} 🧠 Enc: {gn['e']**0.5:.2f}")
                
                # In thông số đo đạc Telemetry PCGrad + PGW (G1)
                if hasattr(self.pcgrad, 'telemetry_data') and self.pcgrad.telemetry_data is not None:
                    td = self.pcgrad.telemetry_data
                    con = td["conflict"]
                    cos_b = td["cosine_before"]
                    cos_a = td["cosine_after"]
                    print(f"    [TELEMETRY] Conflict: L-V:{int(con.get('Lesion,LVO', 0))} L-C:{int(con.get('Lesion,CoW', 0))} V-C:{int(con.get('LVO,CoW', 0))} | "
                          f"CosBefore: L-V:{cos_b.get('Lesion,LVO', 0.0):+.2f} L-C:{cos_b.get('Lesion,CoW', 0.0):+.2f} V-C:{cos_b.get('LVO,CoW', 0.0):+.2f} | "
                          f"CosAfter (Direction Kept): Les:{cos_a.get('Lesion', 1.0):.2f} LVO:{cos_a.get('LVO', 1.0):.2f} CoW:{cos_a.get('CoW', 1.0):.2f}")
                
                enc_str, guide_str = "", ""
                if hasattr(self.pcgrad, '_enc_debug') and self.pcgrad._enc_debug is not None:
                    _ed = self.pcgrad._enc_debug
                    _n, _c = _ed['norms'], _ed['cosine']
                    enc_str = f"| Enc_Norm[L:{_n['Lesion']:.1f} V:{_n['LVO']:.1f} C:{_n['CoW']:.1f}] cos[LV:{_c['L,V']:+.2f} LC:{_c['L,C']:+.2f} VC:{_c['V,C']:+.2f}]"
                    
                raw = self.model.module if hasattr(self.model, "module") else self.model
                if hasattr(raw.decoder, "_lesion_guidance_grad_norm"):
                    g_norm_l = raw.decoder._lesion_guidance_grad_norm
                    if self.scaler is not None: g_norm_l /= self.scaler.get_scale()
                    guide_str += f" L:{g_norm_l:.3f}"
                if hasattr(raw.decoder, "_lvo_guidance_grad_norm"):
                    g_norm = raw.decoder._lvo_guidance_grad_norm
                    if self.scaler is not None: g_norm /= self.scaler.get_scale()
                    guide_str += f" V:{g_norm:.3f}"
                
                if enc_str or guide_str:
                    print(f"    [INFO] Guide_Flow[{guide_str.strip()}] {enc_str}")

            # Bỏ qua Per-task gradient clip thủ công tại đây vì PCGrad đã thực hiện
            # clip per-task 10.0 một cách chính xác trước đó.
            
            # Clip toàn bộ tham số mô hình (global clip)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["total"].item()
            main_loss += losses["main"].item()
            raw_loss += losses.get("unweighted_main", losses["main"].item())
            sum_t_les += losses["l_lesion"]
            sum_t_lvo += losses["l_lvo"]
            sum_t_cow += losses["l_cow"]
            n_batches += 1

        if self.rank == 0:
            # Log spike được theo dõi trực tiếp trong PCGrad nếu cần
            if nan_batches > 0:
                print(f"    [WARN] Đã skip {nan_batches} batches do lỗi NaN/Inf.")
                
            # Compute and print diagnostic stats
            denom = self._diag_lvo_tp + 0.5 * (self._diag_lvo_fp + self._diag_lvo_fn)
            lvo_train_dice = (self._diag_lvo_tp / denom) if denom > 0 else 0.0
            print(f"    [LVO Train] Dice: {lvo_train_dice*100:.2f}% (TP={int(self._diag_lvo_tp)} FP={int(self._diag_lvo_fp)} FN={int(self._diag_lvo_fn)}) | Max_P: {self._diag_lvo_max:.3f}")
            # Reset diagnostics for next epoch
            self._diag_lvo_tp = self._diag_lvo_fp = self._diag_lvo_fn = 0
            self._diag_lvo_max = 0.0
        
        if dist.is_initialized():
            sync = torch.tensor([total_loss, main_loss, float(n_batches), raw_loss, sum_t_les, sum_t_lvo, sum_t_cow], device=self.device)
            dist.all_reduce(sync, op=dist.ReduceOp.SUM)
            avg_loss = sync[0].item() / max(sync[2].item(), 1)
            avg_main = sync[1].item() / max(sync[2].item(), 1)
            avg_raw = sync[3].item() / max(sync[2].item(), 1)
            avg_t_les = sync[4].item() / max(sync[2].item(), 1)
            avg_t_lvo = sync[5].item() / max(sync[2].item(), 1)
            avg_t_cow = sync[6].item() / max(sync[2].item(), 1)
        else:
            avg_loss = total_loss / max(n_batches, 1)
            avg_main = main_loss / max(n_batches, 1)
            avg_raw = raw_loss / max(n_batches, 1)
            avg_t_les = sum_t_les / max(n_batches, 1)
            avg_t_lvo = sum_t_lvo / max(n_batches, 1)
            avg_t_cow = sum_t_cow / max(n_batches, 1)
        return {"train_loss": avg_loss, "train_main": avg_main, "train_raw": avg_raw, "t_les_loss": avg_t_les, "t_lvo_loss": avg_t_lvo, "t_cow_loss": avg_t_cow}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        total_loss, main_loss, raw_loss = 0.0, 0.0, 0.0
        sum_v_les, sum_v_lvo, sum_v_cow = 0.0, 0.0, 0.0
        sum_d_l, sum_d_l_pos, sum_d_c = 0.0, 0.0, 0.0  # [FIX C] sum_d_l_pos: Lesion-positive slice only
        sum_aad, sum_alcd = 0.0, 0.0
        sum_p_v = 0.0
        n_b, n_b_pos = 0, 0  # n_b_pos: số batch có ít nhất 1 Lesion-positive slice
        
        # [FIX] Global LVO stats: gom TP/FP/FN trên toàn Val set thay vì per-batch avg
        lvo_stats = {"tp": 0, "fp": 0, "fn": 0}
        # Patient-level LVO stats
        patient_stats = {}

        # [RAMP] Tính ngưỡng LVO động theo epoch — linear ramp từ thresh_freeze → thresh_unfreeze
        lvo_thr = get_lvo_threshold(epoch + 1, self.metric_weights)  # epoch là 0-indexed, log dùng 1-indexed
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
            main_loss += losses["main"].item()
            raw_loss += losses.get("unweighted_main", losses["main"].item())
            sum_v_les += losses["l_lesion"]
            sum_v_lvo += losses["l_lvo"]
            sum_v_cow += losses["l_cow"]
            sum_p_v += losses.get("p_lvo", 1.0)
            
            # Override thresholds.lvo = lvo_thr (dynamic ramp) cho batch này
            _t = {**self.metric_weights, "thresholds": {**self.metric_weights.get("thresholds", {}), "lvo": lvo_thr}}
            # Truyền lvo_stats và epoch vào để gom TP/FP/FN toàn cục
            max_r = float(self.config.get("loss", {}).get("lvo", {}).get("max_radius", 10.0))
            metrics = compute_all_metrics(preds, lbl, _t, lvo_stats=lvo_stats, epoch=epoch, lvo_max_radius=max_r)

            sum_d_l  += metrics["dice_lesion"]
            sum_d_c  += metrics["dice_cow"]
            sum_aad  += metrics["aad_lesion"]
            sum_alcd += metrics["alcd_lesion"]
            n_b += 1
            # [FIX C] Accumulate Lesion-positive-only Dice (chỉ khi batch có GT Lesion)
            d_l_pos = metrics.get("dice_lesion_pos", None)
            if d_l_pos is not None and d_l_pos < 1.0:  # 1.0 = batch toàn background, bỏ qua
                sum_d_l_pos += d_l_pos
                n_b_pos += 1

            paths = batch.get("path", [""] * inp.shape[0])
            accumulate_patient_lvo_stats(
                preds["lvo"], lbl[:, 1:2], paths, patient_stats,
                threshold=lvo_thr
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

        # [FIX] Đồng bộ TP/FP/FN và D2C distance qua DDP trước khi tính F1
        if dist.is_initialized():
            lvo_tensor = torch.tensor(
                [
                    lvo_stats["tp"], 
                    lvo_stats["fp"], 
                    lvo_stats["fn"],
                    lvo_stats.get("total_dist", 0.0),
                    float(lvo_stats.get("tp_count", 0))
                ], 
                dtype=torch.float32, device=self.device
            )
            dist.all_reduce(lvo_tensor, op=dist.ReduceOp.SUM)
            lvo_stats = {
                "tp": int(lvo_tensor[0].item()), 
                "fp": int(lvo_tensor[1].item()), 
                "fn": int(lvo_tensor[2].item()),
                "total_dist": lvo_tensor[3].item(),
                "tp_count": int(lvo_tensor[4].item())
            }
            
            # Đồng bộ các chỉ số slice bao gồm cả sum_d_l_pos và n_b_pos
            sync = torch.tensor([
                total_loss, main_loss, sum_d_l, sum_d_c, sum_aad, sum_alcd, 
                sum_p_v, float(n_b), raw_loss, sum_d_l_pos, float(n_b_pos),
                sum_v_les, sum_v_lvo, sum_v_cow
            ], device=self.device)
            dist.all_reduce(sync, op=dist.ReduceOp.SUM)
            v = sync.cpu().numpy()
            avg_l, avg_m = v[0]/max(v[7],1), v[1]/max(v[7],1)
            ad_l, ad_c = v[2]/max(v[7],1), v[3]/max(v[7],1)
            a_aad, a_alcd, ap_v = v[4]/max(v[7],1), v[5]/max(v[7],1), v[6]/max(v[7],1)
            avg_raw = v[8]/max(v[7],1)
            ad_l_pos = v[9]/max(v[10], 1)
            avg_v_les = v[11]/max(v[7],1)
            avg_v_lvo = v[12]/max(v[7],1)
            avg_v_cow = v[13]/max(v[7],1)

            # Thu thập và gộp patient_stats từ tất cả các rank
            world_size = dist.get_world_size()
            gathered_stats = [None] * world_size
            dist.all_gather_object(gathered_stats, patient_stats)
            
            merged_stats = {}
            for rank_stats in gathered_stats:
                if rank_stats is None: continue
                for pid, stats in rank_stats.items():
                    if pid not in merged_stats:
                        merged_stats[pid] = {
                            "has_gt": stats["has_gt"],
                            "max_pred": stats["max_pred"]
                        }
                    else:
                        merged_stats[pid]["has_gt"] = merged_stats[pid]["has_gt"] or stats["has_gt"]
                        merged_stats[pid]["max_pred"] = max(merged_stats[pid]["max_pred"], stats["max_pred"])
            patient_stats = merged_stats
        else:
            avg_l = total_loss/max(n_b,1)
            avg_m = main_loss/max(n_b,1)
            avg_raw = raw_loss/max(n_b,1)
            ad_l  = sum_d_l/max(n_b,1)
            ad_c  = sum_d_c/max(n_b,1)
            a_aad = sum_aad/max(n_b,1)
            a_alcd = sum_alcd/max(n_b,1)
            ap_v  = sum_p_v/max(n_b,1)
            ad_l_pos = sum_d_l_pos / max(n_b_pos, 1)
            avg_v_les = sum_v_les / max(n_b, 1)
            avg_v_lvo = sum_v_lvo / max(n_b, 1)
            avg_v_cow = sum_v_cow / max(n_b, 1)

        # Tính patient LVO metrics trên toàn bộ tập dữ liệu đã gộp
        pat = finalize_patient_lvo_acc(
            patient_stats,
            threshold=lvo_thr
        )
        
        # [FIX] Finalize Global LVO Dice
        lvo_dice = finalize_lvo_dice(lvo_stats)
        
        if self.rank == 0:
            # Log LVO Summary (Global D2C + Patient)
            mean_d2c = lvo_stats.get("mean_d2c", 0.0)
            print(f"    [LVO Val] D2C_F1: {lvo_dice:.2f}% (TP={lvo_stats['tp']:.0f} FP={lvo_stats['fp']:.0f} FN={lvo_stats['fn']:.0f} | Mean D2C={mean_d2c:.2f}px) | Pat_Acc: {pat['accuracy']*100:.1f}% (TP={pat['tp']} FP={pat['fp']} FN={pat['fn']} TN={pat['tn']})")
            # Visualize sample tốt nhất (sau khi đã duyệt toàn bộ val loop)
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
        # [FIX] Dùng Slice-level Dice (đã được đồng bộ hoàn hảo qua 2 GPU) thay vì Patient-level (bị lỗi chia cắt DDP)
        slice_dice_lvo = lvo_dice / 100.0
        comp = (w["dice_lesion_weight"] * ad_l + w["dice_lvo_weight"] * slice_dice_lvo + w["dice_cow_weight"] * ad_c)
        
        p_l, p_v, p_c = 1.0, 1.0, 1.0
        if hasattr(self.loss_fn, "current_weights"):
            cw = self.loss_fn.current_weights.tolist()
            p_l, p_v, p_c = cw[0], cw[1], cw[2]

        res = {
            "val_loss": avg_l, "val_main": avg_m, "val_raw": avg_raw, "dice_lesion": ad_l, "dice_lesion_pos": ad_l_pos,
            "dice_lvo": lvo_dice, "dice_cow": ad_c,
            "mean_d2c_lvo": lvo_stats.get("mean_d2c", 0.0),
            "dice_lvo_patient": pat.get("f1", 0.0) * 100.0,
            "p_lesion": p_l,
            "p_lvo": p_v,
            "p_cow": p_c,
            "aad_lesion": a_aad, "alcd_lesion": a_alcd,
            "composite": comp,
            "v_les_loss": avg_v_les, "v_lvo_loss": avg_v_lvo, "v_cow_loss": avg_v_cow
        }

        # Giải phóng thủ công các biến lớn trước khi trả về để tránh giữ tham chiếu vòng trên RAM
        del vis_candidates
        del patient_stats
        if 'gathered_stats' in locals(): del gathered_stats
        if 'merged_stats' in locals(): del merged_stats
        
        return res

    def load_checkpoint(self, path: str):
        if not os.path.exists(path): return 0
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        raw = self.model.module if hasattr(self.model, "module") else self.model
        raw.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and self.optimizer: self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and self.scheduler: self.scheduler.load_state_dict(ckpt["scheduler"])
        if "history" in ckpt: self.history = ckpt["history"]
        return ckpt.get("epoch", 0)

    def _rebuild_train_loader(self, new_file_list: list, epoch: int):
        # Cập nhật danh sách file tại chỗ (in-place) thay vì rebuild loader
        self.train_loader.dataset.update_files(new_file_list)
        if dist.is_initialized() and self.train_loader.sampler is not None:
            self.train_loader.sampler.set_epoch(epoch)

    def fit(self, early_stopping=None, checkpoint=None, start_epoch: int = 0):
        raw = self.model.module if hasattr(self.model, "module") else self.model
        raw.freeze_encoders()
        for epoch in range(start_epoch, self.epochs):
            # [CYCLIC STRIDE] Cập nhật danh sách file huấn luyện và tái cấu trúc DataLoader
            new_train_list = apply_sampling(self.train_files_original, self.config, epoch=epoch)
            self._rebuild_train_loader(new_train_list, epoch)
            
            if epoch == self.freeze_enc_epochs: raw.unfreeze_encoders()
            
            t_m = self.train_one_epoch(epoch)
            import gc
            gc.collect()
            torch.cuda.empty_cache()  # [MEMORY FIX] Giải phóng VRAM rác chống phân mảnh
            
            v_m = self.validate(epoch + 1)
            gc.collect()
            torch.cuda.empty_cache()  # [MEMORY FIX] Giải phóng VRAM rác sau khi validate
            
            if self.rank == 0:
                # Tìm đúng learning rate của Encoder và Decoder dựa vào tên param group
                lr_enc = None
                lr_dec = None
                for g in self.optimizer.param_groups:
                    name = g.get('name', '').lower()
                    if 'encoder' in name:
                        lr_enc = g['lr']
                    elif 'decoder' in name:
                        lr_dec = g['lr']
                
                # Fallback nếu không tìm thấy theo tên
                if lr_enc is None: lr_enc = self.optimizer.param_groups[0]['lr']
                if lr_dec is None: lr_dec = self.optimizer.param_groups[-1]['lr']

                print(f"{'-'*80}\n=> | [Ep {epoch+1:03d}/{self.epochs}] | LR (En/De): {lr_enc:.1e}/{lr_dec:.1e} | Comp: {v_m['composite']:.4f}")
                print(f"   | [VAL] Dice_Lesion: {v_m['dice_lesion']:.4f} (Pos: {v_m['dice_lesion_pos']:.4f}) | Dice_LVO: {v_m['dice_lvo']/100.0:.4f} | Dice_CoW: {v_m['dice_cow']:.4f}")
                print(f"   | [VAL] Loss: {v_m['val_loss']:.4f} (Main: {v_m['val_main']:.4f}, Raw: {v_m['val_raw']:.4f}) | AAD: {v_m['aad_lesion']:.2f}% | ALCD: {v_m['alcd_lesion']:.4f}")
                print(f"   | [TRA] Loss: {t_m['train_loss']:.4f} (Main: {t_m['train_main']:.4f}, Raw: {t_m['train_raw']:.4f})\n{'-'*80}", flush=True)
            self.history.append({**t_m, **v_m, "epoch": epoch + 1})
            if checkpoint and self.rank == 0:
                if (epoch + 1) >= self.config["training"]["checkpoint"].get("start_epoch", 1):
                    checkpoint.update(self.model, self.optimizer, epoch + 1, v_m, scheduler=self.scheduler, history=self.history)
            if early_stopping and (epoch + 1) >= self.config["training"]["early_stopping"].get("start_epoch", 1):
                if early_stopping(v_m["composite"]): break
            self.scheduler.step()
            if hasattr(self.loss_fn, "update_weights_from_metrics"):
                self.loss_fn.update_weights_from_metrics(v_m, epoch)
        return self.history
