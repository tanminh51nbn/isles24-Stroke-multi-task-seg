"""
visualize.py — Trực quan hóa kết quả training và predictions (Bản nâng cấp lâm sàng)
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
    else:
        plt.show()
    plt.close()


# ─── Prediction Overlay (Clinical Style) ──────────────────────────────────────

def overlay_predictions(
    sample: dict,
    preds: dict,
    epoch: int,
    save_dir: str,
    thresholds: dict = None,
):
    """
    Chồng mask dự đoán lên ảnh CTA theo phong cách đối chiếu lâm sàng (GT vs Pred).
    """
    if thresholds is None:
        thresholds = {"lesion": 0.5, "lvo": 0.25, "cow": 0.5}

    os.makedirs(save_dir, exist_ok=True)
    
    # Chuẩn bị dữ liệu
    cta_img = sample["input"][6].cpu().numpy() # Lấy lát cắt CTA trung tâm
    gt_masks = [sample["label"][i].cpu().numpy() for i in range(3)]
    
    pr_masks = [
        (torch.sigmoid(preds["lesion"].squeeze()) > thresholds.get("lesion", 0.5)).float().cpu().numpy(),
        (torch.sigmoid(preds["lvo"].squeeze())    > thresholds.get("lvo", 0.25)).float().cpu().numpy(),
        (torch.sigmoid(preds["cow"].squeeze())    > thresholds.get("cow", 0.5)).float().cpu().numpy(),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Epoch {epoch} — Clinical Review (GT vs AI)", fontsize=14, fontweight="bold")
    
    colors = [(1, 0, 0), (0, 0.5, 1), (0, 1, 0)] # Đỏ (Lesion), Xanh dương (LVO), Xanh lá (CoW)
    task_names = ["Lesion", "LVO", "CoW"]
    
    for i, ax in enumerate(axes):
        ax.imshow(cta_img, cmap="gray")
        ax.set_title("BÁC SĨ (Ground Truth)" if i == 0 else "AI DỰ ĐOÁN")
        
        current_masks = gt_masks if i == 0 else pr_masks
        
        for task_idx in range(3):
            mask = current_masks[task_idx]
            if mask.sum() > 0:
                # Tạo lớp overlay RGBA
                overlay = np.zeros((*mask.shape, 4))
                overlay[..., :3] = colors[task_idx]
                overlay[..., 3] = mask * 0.5 # Độ trong suốt 50%
                ax.imshow(overlay)
        
        ax.axis("off")

    # Thêm chú thích màu sắc (Legend)
    patches = [mpatches.Patch(color=colors[j], label=task_names[j]) for j in range(3)]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=12)

    plt.tight_layout()
    
    # Lưu file
    fname = os.path.basename(sample.get("path", "sample")).replace(".npy", "")
    save_path = os.path.join(save_dir, f"epoch{epoch:03d}_{fname}.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
