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
    [ĐÃ SỬA: INSTANCE-WISE RECALL]
    Tính Tỉ lệ phát hiện (Detection Rate) cho LVO Heatmap thay vì đếm từng Pixel.
    Luật: Cứ đốm sáng của AI "chạm" vào vùng LVO của Bác sĩ thì tính là 1 điểm (Thành công).
    Bắn trượt ra ngoài hoặc không có đốm sáng thì tính là 0 điểm (Thất bại).
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    preds   = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # 1. Kiểm tra xem slice này có bệnh (LVO) không?
    has_lvo = targets.sum(dim=1) > 0

    if has_lvo.sum() == 0:
        # Không có LVO -> Bỏ qua
        return torch.tensor(-1.0)

    # 2. Kiểm tra AI bắn trúng hay trượt
    # (preds * targets) chỉ > 0 khi đốm sáng AI nằm CHỒNG lên vùng bệnh của bác sĩ
    is_hit = ((preds * targets).sum(dim=1) > 0).float()

    # 3. Tính trung bình tỉ lệ phát hiện thành công
    return is_hit[has_lvo].mean()


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
        targets: (B, 3, H, W) — binary masks (hoặc heatmap cho LVO)
        weights: dict composite score weights VÀ thresholds từ train.yaml

    Returns:
        dict {'dice_lesion', 'recall_lvo', 'dice_cow', 'composite'}
    """
    thresholds = weights.get("thresholds", {"lesion": 0.5, "lvo": 0.5, "cow": 0.5})

    # [QUAN TRỌNG] Đối với LVO Heatmap, chúng ta phải nhị phân hóa GT trước khi tính Recall
    # Dùng ngưỡng thấp (0.1) để tạo "vùng đệm" (Hit Zone) lớn hơn, giúp mô hình dễ học hơn.
    target_lvo_bin = (targets[:, 1:2] > 0.1).float()

    d_lesion = dice_score(preds["lesion"],  targets[:, 0:1],   threshold=thresholds["lesion"]).item()
    r_lvo    = recall_score(preds["lvo"],   target_lvo_bin,    threshold=thresholds["lvo"]).item()
    
    # Nếu không có LVO trong batch, giả định recall là 0.5 (trung lập) 
    if r_lvo < 0: r_lvo = 0.5 
    
    d_cow    = dice_score(preds["cow"],     targets[:, 2:3],   threshold=thresholds["cow"]).item()
    comp     = composite_score(d_lesion, r_lvo, d_cow, weights)

    return {
        "dice_lesion": d_lesion,
        "recall_lvo":  r_lvo,
        "dice_cow":    d_cow,
        "composite":   comp,
    }
