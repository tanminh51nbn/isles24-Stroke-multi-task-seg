"""
metrics.py — Clinical Metrics cho Multi-Task Stroke Segmentation

Metrics:
    Lesion: Volumetric Dice — đo độ chồng lấp thể tích ổ nhồi máu
    LVO:    Recall (Sensitivity) — chỉ cần phát hiện được điểm tắc
    CoW:    Volumetric Dice — đo độ chính xác lập bản đồ mạch máu

    Composite = 0.4*Dice_Lesion + 0.4*Recall_LVO + 0.2*Dice_CoW

Lý do dùng Recall cho LVO thay vì Dice:
    LVO là một điểm nhỏ (đôi khi chỉ vài pixel).
    Dice rất nhạy cảm với kích thước → đánh giá không công bằng.
    Recall chỉ hỏi: "Có detect được không?" — đúng với yêu cầu lâm sàng.
"""

import torch


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Tính Dice Score từ raw logits.

    Args:
        logits:    (B, 1, H, W) raw logits
        targets:   (B, 1, H, W) binary targets {0, 1}
        threshold: Ngưỡng để binarize predictions
        smooth:    Epsilon tránh division by zero

    Returns:
        Scalar Dice score (mean over batch)
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    preds   = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)
    return dice.mean()


def recall_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Tính Recall (Sensitivity = TP / (TP + FN)) từ raw logits.
    Dùng cho LVO: "Có bỏ sót điểm tắc mạch nào không?"

    Args:
        logits:    (B, 1, H, W) raw logits
        targets:   (B, 1, H, W) binary targets
        threshold: Ngưỡng binarize

    Returns:
        Scalar Recall score (mean over batch)
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    preds   = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    TP = (preds * targets).sum(dim=1)
    FN = ((1 - preds) * targets).sum(dim=1)

    # Chỉ tính recall cho những slice thực sự có LVO (tránh NaN khi target = 0)
    has_lvo = targets.sum(dim=1) > 0
    recall = (TP + smooth) / (TP + FN + smooth)

    if has_lvo.sum() == 0:
        return torch.tensor(1.0)  # Không có LVO → không có gì để bỏ sót
    return recall[has_lvo].mean()


def composite_score(
    dice_lesion: float,
    recall_lvo: float,
    dice_cow: float,
    weights: dict,
) -> float:
    """
    Tổng hợp các metric thành một điểm số duy nhất.

    Score = w_lesion * Dice_Lesion + w_lvo * Recall_LVO + w_cow * Dice_CoW

    Args:
        dice_lesion: Dice score cho Lesion
        recall_lvo:  Recall score cho LVO
        dice_cow:    Dice score cho CoW
        weights:     dict với keys 'dice_lesion_weight', 'recall_lvo_weight', 'dice_cow_weight'

    Returns:
        Composite score (float, higher is better)
    """
    return (
        weights["dice_lesion_weight"] * dice_lesion +
        weights["recall_lvo_weight"]  * recall_lvo  +
        weights["dice_cow_weight"]    * dice_cow
    )


def compute_all_metrics(preds: dict, targets: torch.Tensor, weights: dict) -> dict:
    """
    Tính toàn bộ metrics trong một lần gọi.

    Args:
        preds:   dict {'lesion', 'lvo', 'cow'} — raw logits
        targets: (B, 3, H, W) — binary masks
        weights: dict composite score weights từ train.yaml

    Returns:
        dict {'dice_lesion', 'recall_lvo', 'dice_cow', 'composite'}
    """
    d_lesion = dice_score(preds["lesion"],  targets[:, 0:1]).item()
    r_lvo    = recall_score(preds["lvo"],   targets[:, 1:2]).item()
    d_cow    = dice_score(preds["cow"],     targets[:, 2:3]).item()
    comp     = composite_score(d_lesion, r_lvo, d_cow, weights)

    return {
        "dice_lesion": d_lesion,
        "recall_lvo":  r_lvo,
        "dice_cow":    d_cow,
        "composite":   comp,
    }
