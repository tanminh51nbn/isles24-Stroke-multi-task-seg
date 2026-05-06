"""
metrics.py — Clinical Metrics chuẩn ISLES'24 ( Kurtlab & Top Leaderboard)

Hệ quy chiếu chung với các đội vô địch:
    1. Dice (%) ↑ : Độ chồng lấp thể tích (Lesion/CoW)
    2. AVD (%) ↓  : Sai lệch thể tích tuyệt đối (Average Volumetric Difference)
    3. F1 (%) ↑   : Chỉ số F1 cho LVO (Instance-level, cân bằng Precision/Recall)
    4. ALCD ↓     : Chênh lệch số lượng ổ tổn thương (Absolute Lesion Count Difference)
"""

import torch
import numpy as np
from scipy.ndimage import label


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    preds = (torch.sigmoid(logits) > threshold).float()
    preds   = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)
    return dice.mean()


def aad_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Average Area Difference (%) — Sai lệch diện tích trên từng lát cắt 2D"""
    preds = (torch.sigmoid(logits) > threshold).float()
    area_p = preds.sum().item()
    area_g = targets.sum().item()
    if area_g == 0:
        return 100.0 if area_p > 0 else 0.0
    return abs(area_p - area_g) / area_g * 100.0


def f1_lvo_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Instance-level F1 Score cho LVO Detection"""
    preds = (torch.sigmoid(logits) > threshold).float().cpu().numpy()
    gt    = (targets > 0.1).float().cpu().numpy() # Dùng 0.1 để binarize heatmap GT
    
    tp, fp, fn = 0, 0, 0
    
    for i in range(preds.shape[0]):
        p_slice = preds[i, 0]
        g_slice = gt[i, 0]
        
        has_p = p_slice.max() > 0
        has_g = g_slice.max() > 0
        
        if has_p and has_g:
            # Kiểm tra xem có chạm nhau không
            if (p_slice * g_slice).sum() > 0: tp += 1
            else: 
                fp += 1
                fn += 1
        elif has_p and not has_g: fp += 1
        elif not has_p and has_g: fn += 1
            
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    return f1 * 100.0


def alcd_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Absolute Lesion Count Difference"""
    preds = (torch.sigmoid(logits) > threshold).float().cpu().numpy()
    gt    = (targets > 0.5).float().cpu().numpy()
    
    total_diff = 0
    for i in range(preds.shape[0]):
        _, n_p = label(preds[i, 0])
        _, n_g = label(gt[i, 0])
        total_diff += abs(n_p - n_g)
    return total_diff / preds.shape[0]


def compute_all_metrics(preds: dict, targets: torch.Tensor, weights: dict) -> dict:
    t = weights.get("thresholds", {"lesion": 0.45, "lvo": 0.05, "cow": 0.5})

    # Lesion Metrics
    d_lesion  = dice_score(preds["lesion"], targets[:, 0:1], threshold=t["lesion"]).item()
    aad_lesion = aad_score(preds["lesion"], targets[:, 0:1], threshold=t["lesion"])
    alcd_lesion = alcd_score(preds["lesion"], targets[:, 0:1], threshold=t["lesion"])

    # LVO Metrics (F1 instead of simple Recall)
    f1_lvo = f1_lvo_score(preds["lvo"], targets[:, 1:2], threshold=t["lvo"])
    
    # CoW Metrics
    d_cow = dice_score(preds["cow"], targets[:, 2:3], threshold=t["cow"]).item()

    # Composite Score (Giữ nguyên logic cũ để so sánh nội bộ, nhưng in thêm chuẩn ISLES)
    w = weights
    comp = (w["dice_lesion_weight"] * d_lesion + 
            w["recall_lvo_weight"]  * (f1_lvo/100.0) + 
            w["dice_cow_weight"]    * d_cow)

    return {
        "dice_lesion": d_lesion,
        "avd_lesion":  avd_lesion,
        "alcd_lesion": alcd_lesion,
        "f1_lvo":      f1_lvo,
        "dice_cow":    d_cow,
        "composite":   comp
    }
