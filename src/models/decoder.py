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
from typing import List, Optional


# ─── Attention Modules ───────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Standard Attention Gate (AG) từ mô hình Attention UNet (Oktay et al.).
    Dùng tín hiệu từ tầng sâu (gating) để lọc không gian tầng nông (skip).
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.GELU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class LightweightDualAttention(nn.Module):
    """
    CBAM-style Dual Attention (Channel + Spatial).
    Tối ưu cho Dual-Encoder (CTA + Perfusion).
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # 1. Channel Attention (Squeeze-and-Excitation cải tiến)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        # 2. Spatial Attention (Conv 7x7)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * out

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.sigmoid(self.spatial_conv(spatial_out))
        return x * spatial_out


class ConvBnGelu(nn.Module):
    """3×3 Conv + BN + GELU — khối cơ bản."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Một cấp decoder:
        1. Upsample 2x (bilinear)
        2. Concatenate skip features từ 2 encoder
        3. 2× ConvBnGelu để học cách kết hợp thông tin
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention_type: Optional[str] = None):
        super().__init__()
        self.attention_type = attention_type
        
        # Cấu hình Attention
        if attention_type == "ag":
            # Gating signal (in_ch) + Skip signal (skip_ch)
            self.ag = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=skip_ch // 2)
        elif attention_type == "dual":
            self.dual_attn = LightweightDualAttention(channels=skip_ch)
        
        # Sau khi kết hợp (Upsampled x + Skip)
        total_in = in_ch + skip_ch
        self.conv1 = ConvBnGelu(total_in, out_ch)
        self.conv2 = ConvBnGelu(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip_cta: torch.Tensor, skip_perf: torch.Tensor) -> torch.Tensor:
        # 1. Upsample feature map từ tầng sâu (Gating signal)
        x_upsampled = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        
        # 2. Chuẩn bị Skip signal (CTA + Perfusion)
        skip = torch.cat([skip_cta, skip_perf], dim=1)
        
        # 3. Áp dụng Attention
        if self.attention_type == "ag":
            # AG dùng x_upsampled để lọc skip
            skip = self.ag(g=x_upsampled, x=skip)
        elif self.attention_type == "dual":
            # Dual Attention tự lọc chính nó dựa trên channel & spatial
            skip = self.dual_attn(skip)
            
        # 4. Nối và xử lý
        x = torch.cat([x_upsampled, skip], dim=1)
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

    def __init__(self, config: dict):
        super().__init__()
        
        # Lấy thông số từ config (model.yaml)
        dec_ch = config["decoder"]["out_channels"] # [512, 256, 128, 64]
        final_ch = config["decoder"].get("final_ch", 16)
        attn_type = config["decoder"].get("attention_type", "dual")

        # Bottleneck: ResNet(2048) + Dense(1024) = 3072 ch
        self.bottleneck = nn.Sequential(
            ConvBnGelu(3072, 1024),
            ConvBnGelu(1024, 1024),
        )

        # skip_ch: Tổng số kênh skip (CTA + Perf) tại mỗi level
        # Level 4: Res(1024) + Dense(512) = 1536
        # Level 3: Res(512)  + Dense(256) = 768
        # Level 2: Res(256)  + Dense(128) = 384
        # Level 1: Res(64)   + Dense(64)  = 128
        
        self.dec4 = DecoderBlock(1024, 1536, dec_ch[0], attn_type)
        self.dec3 = DecoderBlock(dec_ch[0], 768, dec_ch[1], attn_type)
        self.dec2 = DecoderBlock(dec_ch[1], 384, dec_ch[2], attn_type)
        self.dec1 = DecoderBlock(dec_ch[2], 128, dec_ch[3], attn_type)

        # Upsample cuối từ H/2 → H (full resolution) + Refinement
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnGelu(dec_ch[3], final_ch * 2),
            ConvBnGelu(final_ch * 2, final_ch), # Thêm 1 lớp để tăng độ mịn
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
            Tensor (B, final_ch, H, W) — feature map cuối, đưa vào heads
        """
        s1, s2, s3, s4, s5 = cta_skips
        d1, d2, d3, d4, d5 = perf_skips

        # Bottleneck: Concat deepest features
        x = torch.cat([s5, d5], dim=1)  # (B, 2048+1024=3072, H/32, W/32)
        x = self.bottleneck(x)           # (B, 1024, H/32, W/32)

        # Decode từ sâu → nông
        x = self.dec4(x, s4, d4)        # s4=1024, d4=512 -> skip=1536
        x = self.dec3(x, s3, d3)        # s3=512,  d3=256 -> skip=768
        x = self.dec2(x, s2, d2)        # s2=256,  d2=128 -> skip=384
        x = self.dec1(x, s1, d1)        # s1=64,   d1=64  -> skip=128

        x = self.final_upsample(x)      # (B, final_ch, H, W)
        return x
