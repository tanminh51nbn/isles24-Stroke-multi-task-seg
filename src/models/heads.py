"""
heads.py — Multi-Task Segmentation Heads

3 heads độc lập (Lesion, LVO, CoW), mỗi head là:
    [SE Block] → Conv3x3 → BN → ReLU → SpatialDropout2d → Conv1x1 → Raw Logit

[FIX 2] Thêm ChannelAttention (SE Block) trước mỗi Head.
Mỗi Head tự học reweight các kênh feature map:
    - Lesion Head: Học giảm kênh mạch máu, tăng kênh Tmax/thiếu máu
    - CoW Head:    Học tăng kênh CTA cản quang (mạch máu sáng)
    - LVO Head:    Học tập trung vào kênh điểm tắc nghến
Output là raw logits (KHÔNG sigmoid).
BCEWithLogitsLoss và FocalTversky tự tích hợp sigmoid để đảm bảo
numerical stability (tránh log(0)).
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    [FIX 2] Squeeze-and-Excitation Block (SE Block) — per Task Channel Filter.
    
    Mội SegmentationHead được trang bị một SE Block riêng.
    Nó học một bộ trọng số [w1,...,w16] độc lập cho từng task,
    cho phép Lesion Head tự "tắt tai" trước các kênh mạch máu (CoW features)
    và tập trung vào các kênh thiếu tưới máu (Tmax, CBF thấp).
    
    Paper tham khảo: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    """
    def __init__(self, in_ch: int, reduction: int = 4):
        super().__init__()
        mid_ch = max(1, in_ch // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # (B, C, 1, 1)
            nn.Flatten(),                   # (B, C)
            nn.Linear(in_ch, mid_ch),
            nn.ReLU(inplace=True),
            nn.Linear(mid_ch, in_ch),
            nn.Sigmoid(),                   # Trọng số kênh nằm trong [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # w: (B, C) → reshape → (B, C, 1, 1)
        w = self.fc(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w  # Scale channel-wise


# [T2.1] LVO Binary Classification Branch
# Cho model học task dễ hơn trước: "có LVO không?" (binary)
# Signal này dày đặc hơn heatmap loss (BCE trên 1 scalar, không phụ thuộc num_pos)
class LVOClassificationHead(nn.Module):
    """Global Average Pooling → FC → sigmoid → scalar per batch item."""
    def __init__(self, in_ch: int):
        super().__init__()
        mid_ch = max(16, in_ch // 4)
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(in_ch, mid_ch),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(mid_ch, 1),     # (B, 1)
        )
        # Bias init: pi=0.20 → bias = -log((1-0.2)/0.2) = -1.386
        # Những slide có LVO chiếm ~20% tổng số slice
        nn.init.constant_(self.cls[-1].bias, -1.386)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(x)  # (B, 1) raw logit


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
    Cấu trúc: [ChannelAttention] -> ResidualBlock -> Dropout -> Conv1x1
    """

    def __init__(self, in_ch: int, out_ch: int = 1, dropout: float = 0.3):
        super().__init__()
        self.channel_attn = ChannelAttention(in_ch)   # [FIX 2] SE Block riêng cho từng task
        self.res_block = ResidualBlock(in_ch)
        self.dropout   = nn.Dropout2d(p=dropout)
        self.conv_out  = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)   # Bộ lọc kênh task-specific
        x = self.res_block(x)
        x = self.dropout(x)
        return self.conv_out(x)


class LesionClassificationHead(nn.Module):
    """Global Average Pooling → FC → scalar per batch item."""
    def __init__(self, in_ch: int):
        super().__init__()
        mid_ch = max(16, in_ch // 4)
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(in_ch, mid_ch),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(mid_ch, 1),     # (B, 1)
        )
        # Bias init: pi=0.50 → bias = 0.0
        nn.init.constant_(self.cls[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(x)  # (B, 1) raw logit


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

        # [T2.1] LVO Classification Head (binary: có LVO hay không)
        self.lvo_cls_head = LVOClassificationHead(in_ch)
        
        # [NEW] Lesion Classification Head (binary: có Lesion hay không)
        self.lesion_cls_head = LesionClassificationHead(in_ch)

        # [QUAN TRỌNG] Bias Initialization (Chống sụp đổ màn hình)
        # 1. LVO: [T2] Đổi bias -4.595 → -2.0: σ(-2.0)=0.12
        # Bias cũ (σ=0.01) quá conservative, model không tỉnh nổi sau 70 epoch
        nn.init.constant_(self.lvo_head.conv_out.bias, -2.0)
        
        # 2. CoW (Hiếm, mạch mảnh)
        nn.init.constant_(self.cow_head.conv_out.bias, -2.944)
        
        # 3. Lesion (Ít, vùng lõi): pi = 0.05 => bias = -2.944
        nn.init.constant_(self.lesion_head.conv_out.bias, -2.944)

    def forward(self, features: dict) -> dict:
        """
        Args:
            features: Dictionary chứa feature maps từ MultiHeadDecoder
                      {"lesion": Tensor, "lvo": Tensor, "cow": Tensor}
        Returns:
            Dictionary chứa predicted masks/logits + lvo_cls, lesion_cls
        """
        f_lvo = features["lvo"]
        f_lesion = features["lesion"]
        return {
            "lesion":  self.lesion_head(f_lesion),
            "lvo":     self.lvo_head(f_lvo),
            "cow":     self.cow_head(features["cow"]),
            "lvo_cls": self.lvo_cls_head(f_lvo),  # [T2.1] (B, 1) binary LVO cls logit
            "lesion_cls": self.lesion_cls_head(f_lesion),  # [NEW] (B, 1) binary Lesion cls logit
        }
