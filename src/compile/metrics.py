"""
metrics.py — Clinical Metrics chuẩn ISLES'24 ( Kurtlab & Top Leaderboard)

Hệ quy chiếu chung với các đội vô địch:
    1. Dice (%) ↑ : Độ chồng lấp thể tích (Lesion/CoW)
    2. AVD (%) ↓  : Sai lệch thể tích tuyệt đối (Average Volumetric Difference)
    3. F1 (%) ↑   : Chỉ số F1 cho LVO (Instance-level, cân bằng Precision/Recall)
    4. ALCD ↓     : Chênh lệch số lượng ổ tổn thương (Absolute Lesion Count Difference)
"""

import os
import torch
import numpy as np
from scipy.ndimage import label


def get_lvo_threshold(epoch: int, cfg: dict) -> float:
    """
    Tính ngưỡng LVO động theo epoch (Linear Ramp).

    Chiến lược:
      - Epoch ≤ freeze_epoch         : thresh_freeze  (dễ dãi, bắt tín hiệu yếu khi encoder khóa)
      - freeze_epoch < epoch ≤ ramp_end: tăng tuyến tính → thresh_unfreeze
      - Epoch > ramp_end             : thresh_unfreeze (siết chặt, encoder đã hội tụ)

    Args:
        epoch (int): Epoch hiện tại (1-indexed, như in log).
        cfg (dict) : Nhánh ``composite_score`` từ train.yaml.

    Returns:
        float: Ngưỡng LVO hợp lệ cho epoch này.
    """
    ramp_cfg        = cfg.get("lvo_threshold_ramp", {})
    freeze_epoch    = int(ramp_cfg.get("freeze_epoch",    20))
    ramp_end_epoch  = int(ramp_cfg.get("ramp_end_epoch", 30))
    thresh_freeze   = float(ramp_cfg.get("thresh_freeze",   0.10))
    thresh_unfreeze = float(ramp_cfg.get("thresh_unfreeze", 0.30))

    if epoch <= freeze_epoch:
        return thresh_freeze
    elif epoch >= ramp_end_epoch:
        return thresh_unfreeze
    else:
        # Tuyến tính: 0 → 1 trong khoảng (freeze_epoch, ramp_end_epoch]
        t = (epoch - freeze_epoch) / max(ramp_end_epoch - freeze_epoch, 1)
        return thresh_freeze + t * (thresh_unfreeze - thresh_freeze)


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    preds = (torch.sigmoid(logits) > threshold).float()
    preds   = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)
    return dice.mean()


def aad_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Average Area Difference (%) — Tính trung bình sai lệch diện tích trên từng lát cắt.
    Giúp theo dõi độ chính xác về kích thước vùng tổn thương ổn định hơn.
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    
    # Tính diện tích (số pixel) trên từng lát cắt: (B, 1, H, W) -> (B)
    area_p = preds.view(preds.size(0), -1).sum(dim=1)
    area_g = targets.view(targets.size(0), -1).sum(dim=1)
    
    diff_percentages = []
    for p, g in zip(area_p, area_g):
        p_val, g_val = p.item(), g.item()
        if g_val == 0:
            # Nếu thực tế không có tổn thương, dự đoán có (FP) tính là 100% lỗi
            diff_percentages.append(1.0 if p_val > 0 else 0.0)
        else:
            # Tính sai lệch tỉ lệ, giới hạn ở mức 5.0 (500%) để tránh nhiễu epoch đầu làm vọt chỉ số
            diff = abs(p_val - g_val) / g_val
            diff_percentages.append(min(diff, 5.0))
            
    return (sum(diff_percentages) / len(diff_percentages)) * 100.0


def accumulate_lvo_stats(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, lvo_cls: torch.Tensor = None) -> dict:
    """Gom TP/FP/FN từng batch để tính F1 global sau khi duyệt xong Val set."""
    probs = torch.sigmoid(logits)
    if lvo_cls is not None:
        probs_cls = torch.sigmoid(lvo_cls).view(-1, 1, 1, 1)
        probs = probs * probs_cls
    preds = (probs > threshold).float().cpu().numpy()
    gt    = (targets > 0.1).float().cpu().numpy()
    
    tp, fp, fn = 0, 0, 0
    for i in range(preds.shape[0]):
        p_slice = preds[i, 0]
        g_slice = gt[i, 0]
        has_p = p_slice.max() > 0
        has_g = g_slice.max() > 0
        if has_p and has_g:
            if (p_slice * g_slice).sum() > 0: tp += 1
            else: fp += 1; fn += 1
        elif has_p and not has_g: fp += 1
        elif not has_p and has_g: fn += 1
    return {"tp": tp, "fp": fp, "fn": fn}


def finalize_lvo_f1(lvo_stats: dict) -> float:
    """Tính F1 cuối cùng từ TP/FP/FN đã gom trên toàn Val set."""
    tp, fp, fn = lvo_stats["tp"], lvo_stats["fp"], lvo_stats["fn"]
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    return f1 * 100.0


def f1_lvo_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, lvo_cls: torch.Tensor = None) -> float:
    """Instance-level F1 Score cho LVO Detection (per-batch, dùng cho debug)."""
    stats = accumulate_lvo_stats(logits, targets, threshold, lvo_cls)
    return finalize_lvo_f1(stats)


def alcd_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Absolute Lesion Count Difference (slice-level, dùng cho debug nội bộ)"""
    preds = (torch.sigmoid(logits) > threshold).float().cpu().numpy()
    gt    = (targets > 0.5).float().cpu().numpy()
    
    total_diff = 0
    for i in range(preds.shape[0]):
        _, n_p = label(preds[i, 0])
        _, n_g = label(gt[i, 0])
        total_diff += abs(n_p - n_g)
    return total_diff / preds.shape[0]


def accumulate_patient_lesion_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    paths: list,
    patient_lesion_stats: dict,
    threshold: float = 0.5
) -> None:
    """Gom số pixel Lesion dự đoán và GT theo từng bệnh nhân (patient-level).

    Tích lũy across tất cả lát cắt của cùng bệnh nhân để tính AVD và ALCD
    theo chuẩn ISLES'24 (volume-level, không phải slice-level).
    """
    preds = (torch.sigmoid(logits) > threshold).float().cpu().numpy()
    gt    = (targets > 0.5).float().cpu().numpy()

    for i, path in enumerate(paths):
        fname = os.path.basename(path).replace(".npy", "")
        pid   = "_".join(fname.split("_")[:1])   # "sub-stroke0092"

        pred_px = int(preds[i, 0].sum())
        gt_px   = int(gt[i, 0].sum())
        _, n_p  = label(preds[i, 0])
        _, n_g  = label(gt[i, 0])

        if pid not in patient_lesion_stats:
            patient_lesion_stats[pid] = {
                "pred_pixels": pred_px, "gt_pixels": gt_px,
                "pred_components": n_p,  "gt_components": n_g,
            }
        else:
            patient_lesion_stats[pid]["pred_pixels"]     += pred_px
            patient_lesion_stats[pid]["gt_pixels"]       += gt_px
            patient_lesion_stats[pid]["pred_components"] += n_p
            patient_lesion_stats[pid]["gt_components"]   += n_g


def finalize_patient_aad(patient_lesion_stats: dict) -> float:
    """Patient-level Average Volume Difference (%) — tương đương ISLES AVD.

    AVD_patient = |pred_vol - gt_vol| / max(gt_vol, 1) × 100
    Nếu bệnh nhân có GT nhưng pred rỗng: 100% lỗi.
    Nếu bệnh nhân không có GT và pred rỗng: bỏ qua (TN hoàn hảo).
    """
    diffs = []
    for stats in patient_lesion_stats.values():
        gt_v, pred_v = stats["gt_pixels"], stats["pred_pixels"]
        if gt_v == 0 and pred_v == 0:
            continue                          # TN: không tính vào AVD
        elif gt_v == 0:
            diffs.append(100.0)              # FP bệnh nhân: 100% error
        else:
            diffs.append(min(abs(pred_v - gt_v) / gt_v * 100.0, 500.0))
    return sum(diffs) / max(len(diffs), 1)


def finalize_patient_alcd(patient_lesion_stats: dict) -> float:
    """Patient-level Absolute Lesion Count Difference.

    Đếm tổng số thành phần liên thông 2D trên toàn bộ lát cắt của mỗi bệnh nhân
    rồi lấy hiệu tuyệt đối — xấp xỉ 3D ALCD phù hợp cho pipeline 2.5D.
    """
    total_diff = 0
    n = 0
    for stats in patient_lesion_stats.values():
        total_diff += abs(stats["pred_components"] - stats["gt_components"])
        n += 1
    return total_diff / max(n, 1)


def accumulate_patient_lvo_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    paths: list,
    patient_stats: dict,
    threshold: float = 0.5,
    lvo_cls: torch.Tensor = None
) -> None:
    """Gom dự đoán LVO theo bệnh nhân (patient-level) từ batch.

    Mỗi bệnh nhân được định danh bằng patient_id trích từ path.
    patient_stats: dict, cập nhật in-place. Structure:
        {patient_id: {"has_gt": bool, "max_pred": float}}
    """
    probs = torch.sigmoid(logits)
    if lvo_cls is not None:
        probs_cls = torch.sigmoid(lvo_cls).view(-1, 1, 1, 1)
        probs = probs * probs_cls
    preds_prob = probs.float().cpu()
    gt_bin     = (targets > 0.1).float().cpu()

    for i, path in enumerate(paths):
        # Trích patient_id: thường là 2 thành phần đầu tiên của tên file
        # Ví dụ: "sub-r001s001_ses-0001_slice012.npy" → "sub-r001s001"
        fname   = os.path.basename(path).replace(".npy", "")
        pid     = "_".join(fname.split("_")[:1])  # Lấy phần đầu trước dấu _

        has_gt   = gt_bin[i, 0].max().item() > 0
        max_pred = preds_prob[i, 0].max().item()

        if pid not in patient_stats:
            patient_stats[pid] = {"has_gt": has_gt, "max_pred": max_pred}
        else:
            # Một bệnh nhân có nhiều lát cắt: cập nhật max prediction
            patient_stats[pid]["has_gt"]   = patient_stats[pid]["has_gt"] or has_gt
            patient_stats[pid]["max_pred"] = max(
                patient_stats[pid]["max_pred"], max_pred
            )


def finalize_patient_lvo_acc(patient_stats: dict, threshold: float = 0.5) -> dict:
    """Tính Accuracy, TP, FP, FN của LVO detection ở mức bệnh nhân.

    Một bệnh nhân dương tính LVO nếu max_pred trên tất cả lát cắt > threshold.
    Returns: {"accuracy": float, "tp": int, "fp": int, "fn": int, "tn": int, "n": int, "f1": float}
    """
    tp = fp = fn = tn = 0
    for stats in patient_stats.values():
        pred_pos = stats["max_pred"] > threshold
        if stats["has_gt"] and pred_pos:     tp += 1
        elif stats["has_gt"] and not pred_pos: fn += 1
        elif not stats["has_gt"] and pred_pos: fp += 1
        else:                                  tn += 1
    n   = tp + fp + fn + tn
    acc = (tp + tn) / max(n, 1)
    # [METRIC] Patient-level F1
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1_patient = (2 * precision * recall) / (precision + recall + 1e-8)
    # [FIX METRIC] Balanced Accuracy = (Sensitivity + Specificity) / 2
    # Trịt tiêu 2 trivial solutions:
    #   all-positive: Sens=1.0, Spec=0.0 → BalAcc=0.50
    #   all-negative: Sens=0.0, Spec=1.0 → BalAcc=0.50
    sensitivity  = tp / (tp + fn + 1e-8)  # Recall = LVO detection rate
    specificity  = tn / (tn + fp + 1e-8)  # LVO-negative correctly ruled out
    bal_acc      = (sensitivity + specificity) / 2.0
    return {"accuracy": acc, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n, "f1": f1_patient, "bal_acc": bal_acc}


def compute_all_metrics(preds: dict, targets: torch.Tensor, weights: dict, lvo_stats: dict = None) -> dict:
    t = weights.get("thresholds", {"lesion": 0.45, "lvo": 0.05, "cow": 0.5})
    lvo_cls = preds.get("lvo_cls", None)

    # Lesion Metrics
    d_lesion  = dice_score(preds["lesion"], targets[:, 0:1], threshold=t["lesion"]).item()
    aad_lesion = aad_score(preds["lesion"], targets[:, 0:1], threshold=t["lesion"])
    alcd_lesion = alcd_score(preds["lesion"], targets[:, 0:1], threshold=t["lesion"])

    # [FIX C] Dice chỉ trên các slice có GT Lesion > 0 (loại bỏ background-only slice inflate metric)
    # Giá trị này phản ánh đúng khả năng học Lesion thật sự, feed vào PGW thay vì d_lesion
    lesion_gt_flat = targets[:, 0:1].view(targets.size(0), -1)  # (B, H*W)
    has_lesion_gt  = lesion_gt_flat.sum(dim=1) > 0              # (B,) bool mask
    if has_lesion_gt.any():
        d_lesion_pos = dice_score(
            preds["lesion"][has_lesion_gt],
            targets[:, 0:1][has_lesion_gt],
            threshold=t["lesion"]
        ).item()
    else:
        d_lesion_pos = 1.0  # batch này toàn background — không trừng phạt

    # LVO: Gom stats nếu được cung cấp lvo_stats dict (Global mode)
    # Ngược lại tính per-batch như cũ (dùng cho debug)
    if lvo_stats is not None:
        batch_stats = accumulate_lvo_stats(preds["lvo"], targets[:, 1:2], threshold=t["lvo"], lvo_cls=lvo_cls)
        lvo_stats["tp"] += batch_stats["tp"]
        lvo_stats["fp"] += batch_stats["fp"]
        lvo_stats["fn"] += batch_stats["fn"]
        f1_lvo = 0.0  # Sẽ được tính ở cuối epoch bởi finalize_lvo_f1
    else:
        f1_lvo = f1_lvo_score(preds["lvo"], targets[:, 1:2], threshold=t["lvo"], lvo_cls=lvo_cls)
    
    # CoW Metrics
    d_cow = dice_score(preds["cow"], targets[:, 2:3], threshold=t["cow"]).item()

    w = weights
    comp = (w["dice_lesion_weight"] * d_lesion + 
            w["f1_lvo_weight"]      * (f1_lvo/100.0) + 
            w["dice_cow_weight"]    * d_cow)

    return {
        "dice_lesion":     d_lesion,
        "dice_lesion_pos": d_lesion_pos,  # [FIX C] Dice trên Lesion-positive slice — metric thực cho PGW
        "aad_lesion":      aad_lesion,
        "alcd_lesion":     alcd_lesion,
        "f1_lvo":          f1_lvo,
        "dice_cow":        d_cow,
        "composite":       comp
    }
