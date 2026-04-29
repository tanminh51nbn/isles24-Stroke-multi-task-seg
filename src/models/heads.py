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


class ResidualBlock(nn.Module):
    """Khối Residual nhỏ để chuyên môn hóa đặc trưng cho từng task."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.gelu  = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.gelu(out + residual)


class SegmentationHead(nn.Module):
    """
    Đầu phân vùng chuyên gia cho một task.
    Cấu trúc: ResidualBlock -> Dropout -> Conv1x1
    """

    def __init__(self, in_ch: int, out_ch: int = 1, dropout: float = 0.3):
        super().__init__()
        self.res_block = ResidualBlock(in_ch)
        self.dropout   = nn.Dropout2d(p=dropout)
        self.conv_out  = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x)
        x = self.dropout(x)
        return self.conv_out(x)


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
