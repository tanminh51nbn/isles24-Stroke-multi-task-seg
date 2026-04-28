"""
decoder.py — UNet Decoder nhận skip connections từ 2 Encoder

Cấu trúc mỗi DecoderBlock:
    Upsample(2x) → Concat(skip_cta, skip_perfusion) → Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU

Concatenated skip channels tại mỗi level:
    Level 5 (bottleneck): 2048 + 1024 = 3072 ch
    Level 4:              1024 + 512  = 1536 ch
    Level 3:              512  + 256  = 768  ch
    Level 2:              256  + 128  = 384  ch
    Level 1:              64   + 64   = 128  ch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBnRelu(nn.Module):
    """3×3 Conv + BN + ReLU — khối cơ bản."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Một cấp decoder:
        1. Upsample 2x (bilinear)
        2. Concatenate skip features từ 2 encoder
        3. 2× ConvBnRelu để học cách kết hợp thông tin
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        """
        Args:
            in_ch:   Số kênh từ decoder block trước (hoặc bottleneck)
            skip_ch: Số kênh skip đã concatenate (cta_skip + perfusion_skip)
            out_ch:  Số kênh output của block này
        """
        super().__init__()
        # in_ch sau upsample + skip_ch
        total_in = in_ch + skip_ch
        self.conv1 = ConvBnRelu(total_in, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip_cta: torch.Tensor, skip_perf: torch.Tensor) -> torch.Tensor:
        # Upsample lên 2x
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # Concat skip từ 2 encoder
        skip = torch.cat([skip_cta, skip_perf], dim=1)
        # Concat với upsampled feature
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetDecoder(nn.Module):
    """
    UNet Decoder 5 cấp nhận skip features từ Dual-Encoder.

    Chiều sâu đặt theo plan:
        Bottleneck: ResNet(2048) + Dense(1024) = 3072 ch → compress thành 1024 ch
        Level 4: skip(1024+512=1536) + prev(1024) → out 512 ch
        Level 3: skip(512+256=768)   + prev(512)  → out 256 ch
        Level 2: skip(256+128=384)   + prev(256)  → out 128 ch
        Level 1: skip(64+64=128)     + prev(128)  → out 64 ch
        Final: Upsample × 2 → 16 ch (kết nối vào heads)
    """

    def __init__(self):
        super().__init__()
        # Compress bottleneck (3072 → 1024) trước khi decode
        self.bottleneck = nn.Sequential(
            ConvBnRelu(3072, 1024),
            ConvBnRelu(1024, 1024),
        )

        # skip_ch = cta_skip + densenet_skip tại mỗi level (trước Transition)
        # dec4: s4(1024) + d4(1024) = 2048
        # dec3: s3(512)  + d3(512)  = 1024
        # dec2: s2(256)  + d2(256)  = 512
        # dec1: s1(64)   + d1(64)   = 128
        self.dec4 = DecoderBlock(in_ch=1024, skip_ch=2048, out_ch=512)
        self.dec3 = DecoderBlock(in_ch=512,  skip_ch=1024, out_ch=256)
        self.dec2 = DecoderBlock(in_ch=256,  skip_ch=512,  out_ch=128)
        self.dec1 = DecoderBlock(in_ch=128,  skip_ch=128,  out_ch=64)


        # Upsample cuối từ H/2 → H (full resolution) + compress
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnRelu(64, 32),
            ConvBnRelu(32, 16),
        )

    def forward(
        self,
        cta_skips: List[torch.Tensor],
        perf_skips: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            cta_skips:  [s1(64), s2(256), s3(512), s4(1024), s5(2048)] — nông→sâu
            perf_skips: [d1(64), d2(128), d3(256), d4(512),  d5(1024)] — nông→sâu

        Returns:
            Tensor (B, 16, H, W) — feature map cuối, đưa vào heads
        """
        s1, s2, s3, s4, s5 = cta_skips
        d1, d2, d3, d4, d5 = perf_skips

        # Bottleneck: Concat deepest features
        x = torch.cat([s5, d5], dim=1)  # (B, 3072, H/32, W/32)
        x = self.bottleneck(x)           # (B, 1024, H/32, W/32)

        # Decode từ sâu → nông
        x = self.dec4(x, s4, d4)        # (B, 512,  H/16, W/16)
        x = self.dec3(x, s3, d3)        # (B, 256,  H/8,  W/8)
        x = self.dec2(x, s2, d2)        # (B, 128,  H/4,  W/4)
        x = self.dec1(x, s1, d1)        # (B, 64,   H/2,  W/2)

        x = self.final_upsample(x)      # (B, 16,   H,    W)
        return x
