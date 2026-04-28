"""
heads.py — Multi-Task Segmentation Heads

3 heads độc lập (Lesion, LVO, CoW), mỗi head là:
    Conv3x3 → BN → ReLU → SpatialDropout2d → Conv1x1 → Raw Logit

Output là raw logits (KHÔNG sigmoid).
BCEWithLogitsLoss và FocalTversky tự tích hợp sigmoid để đảm bảo
numerical stability (tránh log(0)).
"""

import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    """
    Đầu phân vùng cho một task.

    Kiến trúc:
        Conv 3×3 (in_ch → in_ch) → BN → ReLU
        → SpatialDropout2d (regularization theo channel)
        → Conv 1×1 (in_ch → out_ch)  [raw logit]
    """

    def __init__(self, in_ch: int, out_ch: int = 1, dropout: float = 0.3):
        """
        Args:
            in_ch:   Số kênh đầu vào (từ decoder output, mặc định 16)
            out_ch:  Số kênh đầu ra (1 cho binary segmentation)
            dropout: Xác suất Spatial Dropout (theo kênh, không theo pixel)
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Spatial Dropout2d drop toàn bộ feature map của một kênh
            # Hiệu quả hơn Dropout thông thường cho segmentation (Tompson et al.)
            nn.Dropout2d(p=dropout),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MultiTaskHeads(nn.Module):
    """
    Tập hợp 3 heads độc lập cho Lesion, LVO, CoW.
    Mỗi head có bộ tham số riêng để tối ưu hóa độc lập.
    """

    def __init__(self, in_ch: int = 16, dropout: float = 0.3):
        super().__init__()
        self.lesion_head = SegmentationHead(in_ch, out_ch=1, dropout=dropout)
        self.lvo_head    = SegmentationHead(in_ch, out_ch=1, dropout=dropout)
        self.cow_head    = SegmentationHead(in_ch, out_ch=1, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Tensor (B, in_ch, H, W) từ decoder

        Returns:
            dict với keys 'lesion', 'lvo', 'cow'
            Mỗi value là Tensor (B, 1, H, W) — raw logits
        """
        return {
            "lesion": self.lesion_head(x),
            "lvo":    self.lvo_head(x),
            "cow":    self.cow_head(x),
        }
