"""
visualize.py — Trực quan hóa kết quả training và predictions

Hai nhóm hàm:
    1. Training curves: Loss, Dice, Recall theo epoch
    2. Prediction overlay: Chồng mask dự đoán lên ảnh CTA
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend cho server/Kaggle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional


# ─── Training Curves ──────────────────────────────────────────────────────────

def plot_training_curves(history: List[dict], save_path: Optional[str] = None):
    """
    Vẽ các biểu đồ Loss và Metric theo epoch.

    Args:
        history:   List các dict mỗi epoch: {'epoch', 'train_loss', 'dice_lesion', ...}
        save_path: Nếu cung cấp → lưu file PNG, nếu không → hiển thị
    """
    epochs       = [h["epoch"] for h in history]
    train_losses = [h["train_loss"]   for h in history]
    dice_lesion  = [h["dice_lesion"]  for h in history]
    recall_lvo   = [h["recall_lvo"]   for h in history]
    dice_cow     = [h["dice_cow"]     for h in history]
    composite    = [h["composite"]    for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("ISLES'24 — Training Progress", fontsize=14, fontweight="bold")

    # Panel 1: Loss
    axes[0].plot(epochs, train_losses, color="#E74C3C", linewidth=2, label="Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Per-task Metrics
    axes[1].plot(epochs, dice_lesion, color="#3498DB", linewidth=2, label="Dice Lesion")
    axes[1].plot(epochs, recall_lvo,  color="#E74C3C", linewidth=2, label="Recall LVO",  linestyle="--")
    axes[1].plot(epochs, dice_cow,    color="#2ECC71", linewidth=2, label="Dice CoW",    linestyle=":")
    axes[1].set_title("Per-Task Metrics (Validation)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Composite Score
    axes[2].plot(epochs, composite, color="#9B59B6", linewidth=2.5, label="Composite")
    axes[2].fill_between(epochs, composite, alpha=0.15, color="#9B59B6")
    axes[2].set_title("Composite Score (0.4·Dice_L + 0.4·Recall_LVO + 0.2·Dice_CoW)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].legend()
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualize] Saved training curves: {save_path}")
    else:
        plt.show()
    plt.close()


# ─── Prediction Overlay ───────────────────────────────────────────────────────

def overlay_predictions(
    sample: dict,
    preds: dict,
    epoch: int,
    save_dir: str,
    threshold: float = 0.5,
):
    """
    Chồng mask dự đoán lên ảnh CTA (kênh CTA_w1 của lát cắt trung tâm).

    Layout: CTA Original | GT Masks | Predicted Masks | Overlay

    Args:
        sample:    dict {'input': (18,H,W), 'label': (3,H,W), 'path': str}
        preds:     dict {'lesion': (1,1,H,W), 'lvo': (1,1,H,W), 'cow': (1,1,H,W)} — logits
        epoch:     Epoch hiện tại (dùng trong tên file)
        save_dir:  Thư mục lưu ảnh
        threshold: Ngưỡng binarize predictions
    """
    os.makedirs(save_dir, exist_ok=True)

    inp = sample["input"]  # (18, H, W)
    lbl = sample["label"]  # (3, H, W)

    # Lấy kênh CTA_w1 của lát cắt trung tâm (Z), index = 6
    cta_img = inp[6].cpu().numpy()

    # Ground truth masks
    gt_lesion = lbl[0].cpu().numpy()
    gt_lvo    = lbl[1].cpu().numpy()
    gt_cow    = lbl[2].cpu().numpy()

    # Predicted masks (sigmoid + threshold)
    def to_mask(logit_tensor):
        return (torch.sigmoid(logit_tensor.squeeze()) > threshold).float().cpu().numpy()

    pr_lesion = to_mask(preds["lesion"])
    pr_lvo    = to_mask(preds["lvo"])
    pr_cow    = to_mask(preds["cow"])

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Epoch {epoch} — Prediction Overlay", fontsize=13, fontweight="bold")

    # Row 1: CTA + Ground Truth
    axes[0, 0].imshow(cta_img, cmap="gray")
    axes[0, 0].set_title("CTA (Z-center)")

    axes[0, 1].imshow(cta_img, cmap="gray")
    axes[0, 1].imshow(gt_lesion, alpha=0.5, cmap="Reds")
    axes[0, 1].set_title("GT: Lesion")

    axes[0, 2].imshow(cta_img, cmap="gray")
    axes[0, 2].imshow(gt_lvo, alpha=0.5, cmap="Blues")
    axes[0, 2].set_title("GT: LVO")

    axes[0, 3].imshow(cta_img, cmap="gray")
    axes[0, 3].imshow(gt_cow, alpha=0.5, cmap="Greens")
    axes[0, 3].set_title("GT: CoW")

    # Row 2: Predictions
    axes[1, 0].imshow(cta_img, cmap="gray")
    axes[1, 0].set_title("CTA (reference)")

    axes[1, 1].imshow(cta_img, cmap="gray")
    axes[1, 1].imshow(pr_lesion, alpha=0.5, cmap="Reds")
    axes[1, 1].set_title("Pred: Lesion")

    axes[1, 2].imshow(cta_img, cmap="gray")
    axes[1, 2].imshow(pr_lvo, alpha=0.5, cmap="Blues")
    axes[1, 2].set_title("Pred: LVO")

    axes[1, 3].imshow(cta_img, cmap="gray")
    axes[1, 3].imshow(pr_cow, alpha=0.5, cmap="Greens")
    axes[1, 3].set_title("Pred: CoW")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    fname = os.path.basename(sample.get("path", "sample")).replace(".npy", "")
    save_path = os.path.join(save_dir, f"epoch{epoch:03d}_{fname}.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved overlay: {save_path}")
