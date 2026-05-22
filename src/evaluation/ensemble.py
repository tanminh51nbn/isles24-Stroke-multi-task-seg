"""
ensemble.py — Ensemble Inference cho K-Fold Models

Sau khi chạy xong 5 fold, load tất cả 5 checkpoint và
trung bình kết quả dự đoán (Mean Ensemble) trên tập test.

Mean Ensemble thường tăng +3-5% Dice so với mô hình đơn lẻ
vì mỗi mô hình mắc lỗi ở các vùng khác nhau, và việc
lấy trung bình sẽ "vote" để hủy bỏ các lỗi cục bộ.
"""

import os
import torch
import torch.nn as nn
from typing import List, Dict


def load_ensemble_models(
    model_template: nn.Module,
    fold_dirs: List[str],
    ckpt_name: str = "best_overall.pt",
    device: torch.device = None,
) -> List[nn.Module]:
    """
    Load tất cả checkpoint từ các fold vào danh sách mô hình.

    Args:
        model_template: Mô hình đã khởi tạo (chưa load weights)
        fold_dirs:       Danh sách đường dẫn thư mục checkpoint của từng fold
                         Ví dụ: ["/kaggle/working/outputs/fold_0/checkpoints", ...]
        ckpt_name:       Tên file checkpoint trong mỗi fold
        device:          Device để load model (mặc định CPU nếu None)

    Returns:
        Danh sách mô hình đã load weights, ở chế độ eval
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for fold_idx, fold_dir in enumerate(fold_dirs):
        ckpt_path = os.path.join(fold_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[Ensemble] ⚠ Fold {fold_idx}: Không tìm thấy {ckpt_path}, bỏ qua.")
            continue

        # Clone một instance mới — tránh chia sẻ tham số giữa các mô hình
        import copy
        model = copy.deepcopy(model_template)

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)

        # Xử lý prefix "module." nếu checkpoint được lưu từ DDP
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

        model.to(device)
        model.eval()
        models.append(model)
        print(f"[Ensemble] ✓ Fold {fold_idx}: Loaded {ckpt_path}")

    print(f"[Ensemble] Tổng số mô hình: {len(models)}/{len(fold_dirs)}")
    return models


@torch.no_grad()
def ensemble_predict(
    models: List[nn.Module],
    x: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Mean Ensemble: Trung bình xác suất sigmoid của N mô hình.

    Luật:
        - Mỗi mô hình dự đoán độc lập.
        - Kết quả cuối = trung bình của N xác suất sau sigmoid.
        - Giá trị cuối nằm trong [0, 1] — CÓ THỂ so sánh với threshold trực tiếp.

    Args:
        models: Danh sách N mô hình đã load
        x:      Tensor đầu vào (B, 18, H, W)

    Returns:
        dict {'lesion', 'lvo', 'cow'} — mỗi value là Tensor (B, 1, H, W) prob [0,1]
    """
    if not models:
        raise ValueError("[Ensemble] Không có mô hình nào được load!")

    task_preds = {"lesion": [], "lvo": [], "cow": []}

    for model in models:
        out = model(x)
        task_preds["lesion"].append(torch.sigmoid(out["lesion"]))
        task_preds["cow"].append(torch.sigmoid(out["cow"]))
        
        lvo_prob = torch.sigmoid(out["lvo"])
        if "lvo_cls" in out and out["lvo_cls"] is not None:
            cls_prob = torch.sigmoid(out["lvo_cls"]).view(-1, 1, 1, 1)
            lvo_prob = lvo_prob * cls_prob
        task_preds["lvo"].append(lvo_prob)

    # Trung bình trên dimension mô hình
    return {
        task: torch.stack(preds, dim=0).mean(dim=0)
        for task, preds in task_preds.items()
    }


def build_fold_dirs(output_dir: str, n_folds: int) -> List[str]:
    """Tạo danh sách đường dẫn checkpoint cho từng fold."""
    return [
        os.path.join(output_dir, f"fold_{i}", "checkpoints")
        for i in range(n_folds)
    ]
