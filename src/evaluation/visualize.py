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
    train_raw    = [h.get("train_raw", h["train_loss"]) for h in history]
    val_raw      = [h.get("val_raw", h["val_loss"]) for h in history]
    dice_lesion  = [h["dice_lesion"]  for h in history]
    dice_lesion_pos = [h.get("dice_lesion_pos", 0.0) for h in history]
    dice_lvo       = [h["dice_lvo"]/100.0 for h in history] # Đưa về [0,1] để vẽ chung đồ thị
    dice_cow     = [h["dice_cow"]     for h in history]
    composite    = [h["composite"]    for h in history]

    t_les_loss   = [h.get("t_les_loss", 0.0) for h in history]
    v_les_loss   = [h.get("v_les_loss", 0.0) for h in history]
    t_lvo_loss   = [h.get("t_lvo_loss", 0.0) for h in history]
    v_lvo_loss   = [h.get("v_lvo_loss", 0.0) for h in history]
    t_cow_loss   = [h.get("t_cow_loss", 0.0) for h in history]
    v_cow_loss   = [h.get("v_cow_loss", 0.0) for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("ISLES'24 — Training Progress & Multi-Task Dynamics", fontsize=16, fontweight="bold")
    
    ax_loss = axes[0, 0]
    ax_les = axes[0, 1]
    ax_lvo = axes[0, 2]
    ax_cow = axes[1, 0]
    ax_metrics = axes[1, 1]
    ax_pgw = axes[1, 2]

    ax_loss.plot(epochs, train_losses, color="#E74C3C", linewidth=2, label="Train Loss (Weighted)")
    ax_loss.plot(epochs, val_losses,   color="#3498DB", linewidth=2, label="Val Loss (Weighted)")
    ax_loss.plot(epochs, train_raw,    color="#E74C3C", linewidth=1.5, linestyle="--", alpha=0.6, label="Train Loss (Raw)")
    ax_loss.plot(epochs, val_raw,      color="#3498DB", linewidth=1.5, linestyle="--", alpha=0.6, label="Val Loss (Raw)")
    ax_loss.set_title("1. Overall Loss"); ax_loss.legend(); ax_loss.grid(True, alpha=0.3)

    ax_les.plot(epochs, t_les_loss, color="#E74C3C", linewidth=1.5, label="Train Loss - Lesion", linestyle="--")
    ax_les.plot(epochs, v_les_loss, color="#3498DB", linewidth=2, label="Val Loss - Lesion")
    ax_les.set_title("2. Lesion Loss (Train vs Val)"); ax_les.legend(); ax_les.grid(True, alpha=0.3)

    ax_lvo.plot(epochs, t_lvo_loss, color="#E74C3C", linewidth=1.5, label="Train Loss - LVO", linestyle="--")
    ax_lvo.plot(epochs, v_lvo_loss, color="#3498DB", linewidth=2, label="Val Loss - LVO")
    ax_lvo.set_title("3. LVO Loss (Train vs Val)"); ax_lvo.legend(); ax_lvo.grid(True, alpha=0.3)

    ax_cow.plot(epochs, t_cow_loss, color="#E74C3C", linewidth=1.5, label="Train Loss - CoW", linestyle="--")
    ax_cow.plot(epochs, v_cow_loss, color="#3498DB", linewidth=2, label="Val Loss - CoW")
    ax_cow.set_title("4. CoW Loss (Train vs Val)"); ax_cow.legend(); ax_cow.grid(True, alpha=0.3)

    ax_metrics.plot(epochs, dice_lesion, color="#3498DB", linewidth=2, label="Dice Lesion")
    ax_metrics.plot(epochs, dice_lesion_pos, color="#3498DB", linewidth=2, label="Dice Lesion (Pos)", linestyle="-.")
    ax_metrics.plot(epochs, dice_lvo,       color="#E74C3C", linewidth=2, label="Dice LVO")
    ax_metrics.plot(epochs, dice_cow,    color="#2ECC71", linewidth=2, label="Dice CoW")
    ax_metrics.plot(epochs, composite, color="#9B59B6", linewidth=3.0, label="Composite", alpha=0.8)
    ax_metrics.set_title("5. Validation Metrics"); ax_metrics.legend(); ax_metrics.set_ylim(0, 1); ax_metrics.grid(True, alpha=0.3)

    p_l = [h.get("p_lesion", 1.0) for h in history]
    p_v = [h.get("p_lvo", 1.0) for h in history]
    p_c = [h.get("p_cow", 1.0) for h in history]
    ax_pgw.plot(epochs, p_l, color="#3498DB", linewidth=2, label="P_Lesion")
    ax_pgw.plot(epochs, p_v, color="#E74C3C", linewidth=2, label="P_LVO")
    ax_pgw.plot(epochs, p_c, color="#2ECC71", linewidth=2, label="P_CoW")
    ax_pgw.set_title("6. Competition Weights (PGW)"); ax_pgw.legend(); ax_pgw.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── Smart Sample Selection ───────────────────────────────────────────────────

def select_best_sample(candidates: list) -> dict:
    """Chọn sample tốt nhất trong danh sách ứng viên để visualize.

    Ưu tiên (giảm dần):
        1. Có cả LVO và Lesion — đủ điều kiện so sánh đa chiều
        2. Chỉ có LVO  — mục tiêu quan trọng nhất cần quan sát
        3. Chỉ có Lesion
        4. Không có nhãn nào (fallback)

    Args:
        candidates: List[dict], mỗi phần tử là
            {"input": Tensor, "label": Tensor, "pred": dict, "path": str}
    Returns:
        candidate tốt nhất, hoặc None nếu danh sách rỗng
    """
    if not candidates:
        return None

    def score(c: dict) -> int:
        lbl = c["label"]
        has_lvo    = lbl[1].max().item() > 0.1
        has_lesion = lbl[0].max().item() > 0.5
        if has_lvo and has_lesion: return 3
        if has_lvo:                return 2
        if has_lesion:             return 1
        return 0

    best = max(candidates, key=score)
    best_score = score(best)
    label_desc = {3: "LVO+Lesion", 2: "LVO only", 1: "Lesion only", 0: "No label"}
    print(f"    [VIS] Selected sample: {label_desc.get(best_score, '?')} — {best.get('path', '')}")
    return best


# ─── Prediction Overlay (Dashboard v4) ───────────────────────────────────────

def overlay_predictions(
    sample: dict,
    preds: dict,
    title_prefix: str = "Inference",
    file_prefix: str = "clinical",
    save_dir: Optional[str] = None,
    thresholds: dict = None,
    show: bool = False
):
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

    # Lấy trung bình các kênh để hiển thị cấu trúc đầy đủ nhất
    cta_img  = _norm(sample["input"][0:6].mean(dim=0))  # Mean của 6 kênh CTA
    perf_img = _norm(sample["input"][6:18].mean(dim=0)) # Mean của 12 kênh Perfusion

    # GT
    gt_lesion = sample["label"][0].cpu().numpy()
    gt_lvo    = sample["label"][1].cpu().numpy()
    gt_cow    = sample["label"][2].cpu().numpy()

    # Pred
    lvo_t = thresholds.get("lvo", 0.05)
    sig_lesion_t = torch.sigmoid(preds["lesion"].squeeze()).float().cpu()
    sig_lesion = sig_lesion_t.numpy()
    
    sig_lvo_t = torch.sigmoid(preds["lvo"].squeeze()).float().cpu()
    sig_lvo = sig_lvo_t.numpy()
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

    # 3. Layout 2x3
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    title = f"{title_prefix} | Dice_L: {dice_l:.4f} | {lvo_msg}"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # --- Hàng 1: CTA, Nhãn gốc, AI phân vùng ---
    axes[0, 0].imshow(cta_img, cmap="bone"); axes[0, 0].set_title("CTA"); axes[0, 0].axis("off")
    
    # Nhãn gốc (GT Overlay trên CTA)
    axes[0, 1].imshow(cta_img, cmap="bone")
    gt_rgba = np.zeros((*gt_lesion.shape, 4))
    gt_rgba[gt_lesion > 0] = [0, 0, 0.8, 0.5]       # Xanh dương: Lesion (50% opacity)
    gt_rgba[gt_cow > 0] = [0, 0.7, 0, 0.5]          # Xanh lá: CoW (50% opacity)
    gt_rgba[gt_lvo > 0.1] = [1.0, 0, 0, 1.0]        # Đỏ: LVO (100% opacity, đè lên trên)
    axes[0, 1].imshow(gt_rgba)
    axes[0, 1].set_title("Nhãn gốc"); axes[0, 1].axis("off")

    # AI phân vùng (Pred Overlay trên CTA)
    axes[0, 2].imshow(cta_img, cmap="bone")
    pr_rgba = np.zeros((*pr_lesion_bin.shape, 4))
    pr_rgba[pr_lesion_bin > 0] = [0, 0, 0.8, 0.5]
    pr_rgba[pr_cow_bin > 0] = [0, 0.7, 0, 0.5]
    axes[0, 2].imshow(pr_rgba)
    
    # LVO Heatmap (Hot) in AI phân vùng
    if sig_lvo.max() > 0:
        sig_lvo_viz = sig_lvo.copy()
        sig_lvo_viz[sig_lvo_viz < 0.01] = np.nan  # Giữ lại nền tối để heatmap trông đậm đà hơn
        axes[0, 2].imshow(sig_lvo_viz, cmap="hot", alpha=0.6, vmin=0, vmax=1)
        
    axes[0, 2].set_title("AI phân vùng"); axes[0, 2].axis("off")

    # --- Hàng 2: Perfusion, LVO Heatmap (AI), Lesion Heatmap (AI) ---
    axes[1, 0].imshow(perf_img, cmap="inferno"); axes[1, 0].set_title("Perfusion"); axes[1, 0].axis("off")

    # LVO Heatmap (AI)
    axes[1, 1].imshow(cta_img, cmap="bone")
    if sig_lvo.max() > 0:
        axes[1, 1].imshow(sig_lvo, cmap="hot", alpha=0.6, vmin=0, vmax=1)
    axes[1, 1].set_title("LVO Heatmap (AI)"); axes[1, 1].axis("off")

    # Lesion Error Map (AI)
    axes[1, 2].imshow(cta_img, cmap="bone")
    
    # Tính toán TP, FP, FN cho Lesion
    err_map = np.zeros((*pr_lesion_bin.shape, 3))
    tp_mask = (pr_lesion_bin == 1) & (gt_lesion == 1)
    fp_mask = (pr_lesion_bin == 1) & (gt_lesion == 0)
    fn_mask = (pr_lesion_bin == 0) & (gt_lesion == 1)
    
    err_map[..., 0] = fp_mask * 1.0 # Báo ảo (FP) -> Đỏ
    err_map[..., 1] = tp_mask * 1.0 # Bắt đúng (TP) -> Xanh lục
    err_map[..., 2] = fn_mask * 1.0 # Bỏ sót (FN) -> Xanh dương
    
    alpha_mask = (tp_mask | fp_mask | fn_mask).astype(float) * 0.7
    
    axes[1, 2].imshow(err_map, alpha=alpha_mask)
    axes[1, 2].set_title("Lesion Error Map (FP: Đỏ, FN: Xanh, TP: Lục)"); axes[1, 2].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=(0, 0, 0.8), label="Lesion"),
        mpatches.Patch(color=(0, 0.7, 0), label="CoW"),
        mpatches.Patch(color=(1.0, 0, 0), label="LVO")
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=12)

    if save_dir:
        fname = os.path.basename(sample.get("path", "sample")).replace(".npy", "")
        save_path = os.path.join(save_dir, f"{file_prefix}_{fname}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    if show: plt.show()
    plt.close()

