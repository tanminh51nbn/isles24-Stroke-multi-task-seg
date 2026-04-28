"""
callbacks.py — Early Stopping và Model Checkpoint

EarlyStopping:
    Theo dõi Composite Score (metric tổng hợp).
    Dừng training khi không cải thiện sau N epoch liên tiếp.

ModelCheckpoint:
    Lưu 3 phiên bản checkpoint cho 3 use-case lâm sàng:
    1. best_overall.pt  → Composite Score cao nhất (mô hình sản xuất)
    2. best_lesion.pt   → Dice Lesion cao nhất (đo thể tích ổ nhồi máu)
    3. best_lvo.pt      → Recall LVO cao nhất (cấp cứu phát hiện tắc mạch)
"""

import os
import torch


class EarlyStopping:
    """
    Dừng training khi metric không cải thiện sau `patience` epoch.

    Args:
        patience:   Số epoch chờ không cải thiện trước khi dừng
        min_delta:  Ngưỡng cải thiện tối thiểu để tính là "có tiến bộ"
        mode:       'max' (metric cao hơn là tốt hơn, ví dụ Dice)
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Args:
            score: Metric hiện tại (Composite Score)

        Returns:
            True nếu nên dừng training
        """
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score > self.best_score + self.min_delta) if self.mode == "max" \
                   else (score < self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] Không cải thiện {self.counter}/{self.patience} epoch")
            if self.counter >= self.patience:
                self.should_stop = True
                print("[EarlyStopping] DỪNG TRAINING!")

        return self.should_stop


class ModelCheckpoint:
    """
    Lưu checkpoint tốt nhất cho từng metric lâm sàng.

    Args:
        save_dir: Thư mục lưu checkpoint
        config:   Dict từ train.yaml (phần checkpoint)
    """

    def __init__(self, save_dir: str, config: dict):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        ckpt_cfg = config["training"]["checkpoint"]
        self.save_overall = ckpt_cfg.get("save_best_overall", True)
        self.save_lesion  = ckpt_cfg.get("save_best_lesion",  True)
        self.save_lvo     = ckpt_cfg.get("save_best_lvo",     True)

        self.best_composite = -float("inf")
        self.best_lesion    = -float("inf")
        self.best_lvo       = -float("inf")

    def update(self, model, optimizer, epoch: int, metrics: dict):
        """
        Kiểm tra và lưu checkpoint nếu metric tốt hơn.

        Args:
            model:     DualEncoderUNet (hoặc DDP-wrapped)
            optimizer: AdamW optimizer
            epoch:     Epoch hiện tại
            metrics:   dict {'composite', 'dice_lesion', 'recall_lvo', 'dice_cow'}
        """
        # Unwrap DDP nếu cần
        raw_model = model.module if hasattr(model, "module") else model

        def _save(path, note):
            torch.save({
                "epoch":      epoch,
                "model":      raw_model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "metrics":    metrics,
            }, path)
            print(f"[Checkpoint] Saved {note}: {path}")

        comp    = metrics["composite"]
        lesion  = metrics["dice_lesion"]
        lvo     = metrics["recall_lvo"]

        if self.save_overall and comp > self.best_composite:
            self.best_composite = comp
            _save(os.path.join(self.save_dir, "best_overall.pt"), f"Overall (composite={comp:.4f})")

        if self.save_lesion and lesion > self.best_lesion:
            self.best_lesion = lesion
            _save(os.path.join(self.save_dir, "best_lesion.pt"), f"Lesion (dice={lesion:.4f})")

        if self.save_lvo and lvo > self.best_lvo:
            self.best_lvo = lvo
            _save(os.path.join(self.save_dir, "best_lvo.pt"), f"LVO (recall={lvo:.4f})")
