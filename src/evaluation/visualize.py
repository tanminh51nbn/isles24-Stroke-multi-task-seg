"""
visualize.py — Trực quan hóa kết quả (Bản "LVO Sát Thủ" v4)

Nâng cấp: 
    - Hiển thị 2 hàng 4 cột (Dashboard Lâm sàng).
    - Bao gồm cả CTA và Perfusion để đối chiếu.
    - Tách biệt các thành phần dự đoán để soi chi tiết.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional


# ─── Training Curves ──────────────────────────────────────────────────────────

def plot_training_curves(history: List[dict], save_path: Optional[str] = None):
    epochs       = [h["epoch"] for h in history]
    train_losses = [h["train_loss"]   for h in history]
    val_losses   = [h["val_loss"]     for h in history]
    dice_lesion  = [h["dice_lesion"]  for h in history]
    f1_lvo       = [h["f1_lvo"]/100.0 for h in history] # Đưa về [0,1] để vẽ chung đồ thị
    dice_cow     = [h["dice_cow"]     for h in history]
    composite    = [h["composite"]    for h in history]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("ISLES'24 — Training Progress & Multi-Task Dynamics", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, train_losses, color="#E74C3C", linewidth=2, label="Train Loss")
    axes[0].plot(epochs, val_losses,   color="#3498DB", linewidth=2, label="Val Loss", linestyle="--")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, dice_lesion, color="#3498DB", linewidth=2, label="Dice Lesion")
    axes[1].plot(epochs, f1_lvo,       color="#E74C3C", linewidth=2, label="F1 LVO",  linestyle="--")
    axes[1].plot(epochs, dice_cow,    color="#2ECC71", linewidth=2, label="Dice CoW",    linestyle=":")
    axes[1].set_title("Validation Metrics"); axes[1].legend(); axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)

    p_l = [h.get("p_lesion", 1.0) for h in history]
    p_v = [h.get("p_lvo", 1.0) for h in history]
    p_c = [h.get("p_cow", 1.0) for h in history]
    axes[2].plot(epochs, p_l, color="#3498DB", alpha=0.8, label="P_Lesion")
    axes[2].plot(epochs, p_v, color="#E74C3C", alpha=0.8, label="P_LVO")
    axes[2].plot(epochs, p_c, color="#2ECC71", alpha=0.8, label="P_CoW")
    axes[2].set_title("Competition Weights (P)"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    axes[3].plot(epochs, composite, color="#9B59B6", linewidth=2.5, label="Composite")
    axes[3].fill_between(epochs, composite, alpha=0.15, color="#9B59B6")
    axes[3].set_title("Overall Performance"); axes[3].legend(); axes[3].set_ylim(0, 1); axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── Prediction Overlay (Dashboard v4) ───────────────────────────────────────

def overlay_predictions(sample: dict, preds: dict, epoch: int, save_dir: Optional[str] = None, thresholds: dict = None, show: bool = False):
    if thresholds is None:
        thresholds = {"lesion": 0.45, "lvo": 0.05, "cow": 0.5}

    if save_dir: os.makedirs(save_dir, exist_ok=True)

    def _norm(img: torch.Tensor) -> np.ndarray:
        """Chuẩn hóa ảnh về [0, 255] để hiển thị, xử lý clipping cho CTA/Tmax."""
        arr = img.detach().cpu().numpy().astype(np.float32)
        # Loại bỏ các giá trị outlier cực đoan
        p99 = np.percentile(arr, 99.9)
        arr = np.clip(arr, 0, p99)
        # Min-max normalization
        amin, amax = arr.min(), arr.max()
        if amax > amin:
            arr = (arr - amin) / (amax - amin)
        return (arr * 255).astype(np.uint8)

    cta_img  = _norm(sample["input"][6]) # CTA lát cắt trung tâm
    perf_img = _norm(sample["input"][7]) # Tmax lát cắt trung tâm (Physiology)

    # GT
    gt_lesion = sample["label"][0].cpu().numpy()
    gt_lvo    = sample["label"][1].cpu().numpy()
    gt_cow    = sample["label"][2].cpu().numpy()

    # Pred
    lvo_t = thresholds.get("lvo", 0.05)
    sig_lesion = torch.sigmoid(preds["lesion"].squeeze()).float().cpu().numpy()
    sig_lvo    = torch.sigmoid(preds["lvo"].squeeze()).float().cpu().numpy()
    sig_cow    = torch.sigmoid(preds["cow"].squeeze()).float().cpu().numpy()

    pr_lesion_bin = (sig_lesion > thresholds.get("lesion", 0.45)).astype(float)
    pr_lvo_bin    = (sig_lvo > lvo_t).astype(float)
    pr_cow_bin    = (sig_cow > thresholds.get("cow", 0.5)).astype(float)

    # 2. Tính toán Metric nhanh
    intersection_l = (pr_lesion_bin * gt_lesion).sum()
    dice_l = (2. * intersection_l) / (pr_lesion_bin.sum() + gt_lesion.sum() + 1e-8)
    
    # F1-LVO & Status
    has_gt_lvo = gt_lvo.max() > 0.1
    has_pr_lvo = pr_lvo_bin.max() > 0
    if has_gt_lvo and has_pr_lvo:
        lvo_msg = "LVO: HIT (Đã bắt)" if (pr_lvo_bin * (gt_lvo > 0.1)).sum() > 0 else "LVO: FALSE ALARM (Nhầm)"
    elif not has_gt_lvo and has_pr_lvo:
        lvo_msg = "LVO: FALSE POSITIVE (Báo ảo)"
    elif has_gt_lvo and not has_pr_lvo:
        lvo_msg = "LVO: MISS (Bỏ sót)"
    else:
        lvo_msg = "LVO: N/A (Sạch)"

    # 3. Layout 2x4
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    title = f"Epoch {epoch} | Sample Dice_L: {dice_l:.4f} | {lvo_msg}"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # --- Hàng 1: Tham chiếu & Tổng hợp ---
    axes[0, 0].imshow(cta_img, cmap="bone"); axes[0, 0].set_title("CTA (Anatomy Map)"); axes[0, 0].axis("off")
    axes[0, 1].imshow(perf_img, cmap="inferno"); axes[0, 1].set_title("Perfusion (Physiology)"); axes[0, 1].axis("off")
    
    # [1.3] GT Overlay
    axes[0, 2].imshow(cta_img, cmap="bone")
    gt_overlay = np.zeros((*gt_lesion.shape, 3))
    gt_overlay[..., 0] = gt_lesion * 0.7  # Đỏ: Lesion
    gt_overlay[..., 1] = gt_cow * 0.4     # Xanh lá: CoW
    gt_overlay[..., 2] = (gt_lvo > 0.1).astype(float) * 0.9 # Xanh dương: LVO (Hạ ngưỡng để thấy rõ)
    axes[0, 2].imshow(gt_overlay, alpha=0.5)
    axes[0, 2].set_title("BÁC SĨ (Ground Truth)"); axes[0, 2].axis("off")

    # [1.4] Pred Overlay
    axes[0, 3].imshow(cta_img, cmap="bone")
    pr_overlay = np.zeros((*pr_lesion_bin.shape, 3))
    pr_overlay[..., 0] = pr_lesion_bin * 0.7
    pr_overlay[..., 1] = pr_cow_bin * 0.4
    pr_overlay[..., 2] = pr_lvo_bin * 0.9
    axes[0, 3].imshow(pr_overlay, alpha=0.5)
    axes[0, 3].set_title("AI DỰ ĐOÁN (Tổng hợp)"); axes[0, 3].axis("off")

    # --- Hàng 2: Chi tiết từng Task ---
    axes[1, 0].imshow(sig_lesion, cmap="Reds", vmin=0, vmax=1)
    axes[1, 0].set_title("Lesion Probability Map"); axes[1, 0].axis("off")

    # [2.2] LVO Heatmap (Dự đoán)
    axes[1, 1].imshow(cta_img, cmap="bone")
    axes[1, 1].imshow(sig_lvo, cmap="jet", alpha=0.6, vmin=0, vmax=1)
    axes[1, 1].set_title("LVO Heatmap (AI Prediction)"); axes[1, 1].axis("off")

    axes[1, 2].imshow(sig_cow, cmap="Greens", vmin=0, vmax=1)
    axes[1, 2].set_title("CoW Anatomy Map"); axes[1, 2].axis("off")

    error_map = np.abs(pr_lesion_bin - gt_lesion)
    axes[1, 3].imshow(error_map, cmap="magma")
    axes[1, 3].set_title("Lesion Error Map"); axes[1, 3].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=(0.7, 0, 0), label="Lesion"),
        mpatches.Patch(color=(0, 0.4, 0), label="CoW"),
        mpatches.Patch(color=(0, 0, 0.9), label="LVO")
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=12)

    if save_dir:
        fname = os.path.basename(sample.get("path", "sample")).replace(".npy", "")
        save_path = os.path.join(save_dir, f"epoch{epoch:03d}_{fname}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    if show: plt.show()
    plt.close()
