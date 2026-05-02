"""
visualize.py — Trực quan hóa kết quả training và predictions (Bản nâng cấp lâm sàng v3)

Thay đổi v3:
    - LVO được hiển thị dạng Heatmap (colormap 'hot') thay vì overlay màu đơn.
      Điều này phản ánh đúng bản chất Gaussian Heatmap của nhãn LVO mới.
    - GT LVO: Heatmap gốc từ dataset (giá trị [0,1] liên tục).
    - Pred LVO: Sigmoid của logit LVO head (giá trị [0,1] liên tục).
"""

import os
import numpy as np
import torch
import matplotlib
# matplotlib.use("Agg")  # Sẽ tự động nhận diện backend tùy môi trường
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional


# ─── Training Curves ──────────────────────────────────────────────────────────

def plot_training_curves(history: List[dict], save_path: Optional[str] = None):
    """Vẽ các biểu đồ Loss và Metric theo epoch."""
    epochs       = [h["epoch"] for h in history]
    train_losses = [h["train_loss"]   for h in history]
    val_losses   = [h["val_loss"]     for h in history]
    dice_lesion  = [h["dice_lesion"]  for h in history]
    recall_lvo   = [h["recall_lvo"]   for h in history]
    dice_cow     = [h["dice_cow"]     for h in history]
    composite    = [h["composite"]    for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("ISLES'24 — Training Progress", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, train_losses, color="#E74C3C", linewidth=2, label="Train Loss")
    axes[0].plot(epochs, val_losses,   color="#3498DB", linewidth=2, label="Val Loss", linestyle="--")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, dice_lesion, color="#3498DB", linewidth=2, label="Dice Lesion")
    axes[1].plot(epochs, recall_lvo,  color="#E74C3C", linewidth=2, label="Recall LVO",  linestyle="--")
    axes[1].plot(epochs, dice_cow,    color="#2ECC71", linewidth=2, label="Dice CoW",    linestyle=":")
    axes[1].set_title("Per-Task Metrics (Validation)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend(); axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, composite, color="#9B59B6", linewidth=2.5, label="Composite")
    axes[2].fill_between(epochs, composite, alpha=0.15, color="#9B59B6")
    axes[2].set_title("Composite Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].legend(); axes[2].set_ylim(0, 1); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── Prediction Overlay (Clinical Style v3) ───────────────────────────────────

def overlay_predictions(
    sample: dict,
    preds: dict,
    epoch: int,
    save_dir: Optional[str] = None,
    thresholds: dict = None,
    show: bool = False,
):
    """
    Hiển thị kết quả phân vùng theo phong cách đối chiếu lâm sàng (GT vs Pred).

    Cách hiển thị:
        - Lesion: Overlay màu đỏ (mask nhị phân).
        - LVO:    Heatmap colormap 'hot' (giá trị liên tục [0,1]).
                  Hiển thị riêng hàng dưới để thấy rõ quầng sáng Gaussian.
        - CoW:    Overlay màu xanh lá (mask nhị phân).
    """
    if thresholds is None:
        thresholds = {"lesion": 0.5, "lvo": 0.15, "cow": 0.5}

    if save_dir: os.makedirs(save_dir, exist_ok=True)

    # ── Chuẩn bị dữ liệu ──────────────────────────────────────────────────────
    cta_img = sample["input"][6].cpu().numpy()

    # GT: LVO là Heatmap liên tục [0,1], Lesion/CoW là mask nhị phân
    gt_lesion = sample["label"][0].cpu().numpy()
    gt_lvo    = sample["label"][1].cpu().numpy()          # Heatmap [0,1]
    gt_cow    = sample["label"][2].cpu().numpy()

    # Pred: LVO là sigmoid của logit (liên tục), Lesion/CoW dùng threshold
    lvo_thresh = thresholds.get("lvo", 0.15)
    pr_lesion  = (torch.sigmoid(preds["lesion"].squeeze()) > thresholds.get("lesion", 0.5)).float().cpu().numpy()
    pr_lvo     = torch.sigmoid(preds["lvo"].squeeze()).float().cpu().numpy()   # Heatmap [0,1]
    pr_lvo_bin = (pr_lvo > lvo_thresh).astype(float)                          # Binary để overlay
    pr_cow     = (torch.sigmoid(preds["cow"].squeeze()) > thresholds.get("cow", 0.5)).float().cpu().numpy()

    # ── Layout 2x2: [GT | Pred] × [Tổng quan | LVO Heatmap] ──────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    title = f"Epoch {epoch} — Clinical Review" if epoch < 999 else "Final Clinical Review"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Hàng trên: Overlay tổng quan (Lesion đỏ + LVO xanh dương + CoW xanh lá) ──
    for col, (lesion, cow, lvo_bin, panel_title) in enumerate([
        (gt_lesion, gt_cow, (gt_lvo > lvo_thresh).astype(float), "BÁC SĨ (Ground Truth)"),
        (pr_lesion, pr_cow, pr_lvo_bin,                           "AI DỰ ĐOÁN"),
    ]):
        ax = axes[0, col]
        ax.imshow(cta_img, cmap="bone")
        ax.set_title(panel_title, fontsize=12, fontweight="bold")

        # Lesion = đỏ
        if lesion.sum() > 0:
            overlay = np.zeros((*lesion.shape, 4))
            overlay[..., 0] = 1.0
            overlay[..., 3] = lesion * 0.5
            ax.imshow(overlay)

        # LVO detected = xanh dương
        if lvo_bin.sum() > 0:
            overlay = np.zeros((*lvo_bin.shape, 4))
            overlay[..., 2] = 1.0
            overlay[..., 3] = lvo_bin * 0.65
            ax.imshow(overlay)

        # CoW = xanh lá
        if cow.sum() > 0:
            overlay = np.zeros((*cow.shape, 4))
            overlay[..., 1] = 1.0
            overlay[..., 3] = cow * 0.4
            ax.imshow(overlay)

        ax.axis("off")

    # ── Hàng dưới: LVO Heatmap riêng biệt — nhìn rõ quầng sáng Gaussian ──────
    for col, (lvo_heat, heat_title) in enumerate([
        (gt_lvo, "LVO Heatmap — BÁC SĨ"),
        (pr_lvo, "LVO Heatmap — AI DỰ ĐOÁN"),
    ]):
        ax = axes[1, col]
        ax.imshow(cta_img, cmap="bone")
        if lvo_heat.max() > 0:
            # colormap 'hot': đen → đỏ → cam → vàng → trắng (đỉnh 1.0)
            ax.imshow(lvo_heat, cmap="hot", alpha=0.65, vmin=0, vmax=1)
        ax.set_title(heat_title, fontsize=11)
        ax.axis("off")

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=(1, 0, 0), label="Lesion"),
        mpatches.Patch(color=(0, 0, 1), label="LVO (detected)"),
        mpatches.Patch(color=(0, 1, 0), label="CoW"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=11)
    plt.tight_layout()

    if save_dir:
        fname = os.path.basename(sample.get("path", "sample")).replace(".npy", "")
        save_path = os.path.join(save_dir, f"epoch{epoch:03d}_{fname}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()
