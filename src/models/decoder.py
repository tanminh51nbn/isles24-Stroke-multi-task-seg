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
    Thiết kế "Kính lọc thông minh" để lọc đặc trưng CTA + Perfusion.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # 1. Channel Attention
        # Kết hợp MaxPool (Saliency) và AvgPool (Context)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # 2. Spatial Attention
        # Conv 7x7 để có tầm nhìn rộng, bao quát được các mạch máu dài
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Bước 1: Lọc Channel
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_weight = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_weight

        # Bước 2: Lọc Spatial
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        spatial_mask = torch.cat([avg_mask, max_mask], dim=1)
        spatial_weight = self.sigmoid(self.spatial_conv(spatial_mask))
        
        return x * spatial_weight


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
        1. Upsample 2x
        2. Tích hợp Attention (AG hoặc Dual CBAM)
        3. Kết hợp Skip features từ CTA & Perfusion
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention_type: Optional[str] = None):
        super().__init__()
        self.attention_type = attention_type
        
        # Cấu hình Attention lên nhánh Skip
        if attention_type == "ag":
            self.ag = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=skip_ch // 2)
        elif attention_type == "dual":
            self.dual_attn = LightweightDualAttention(channels=skip_ch)
        
        # Sau khi concat (Upsampled + Skip)
        self.conv1 = ConvBnGelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBnGelu(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip_cta: torch.Tensor, skip_perf: torch.Tensor) -> torch.Tensor:
        # 1. Upsample signal từ tầng sâu
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        
        # 2. Kết hợp Skip CTA & Perfusion
        skip = torch.cat([skip_cta, skip_perf], dim=1)
        
        # 3. Lọc thông tin qua Attention
        if self.attention_type == "ag":
            skip = self.ag(g=x_up, x=skip)
        elif self.attention_type == "dual":
            skip = self.dual_attn(skip)
            
        # 4. Concat và xử lý đặc trưng kết hợp
        out = torch.cat([x_up, skip], dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class UNetDecoder(nn.Module):
    """
    UNet Decoder nhận skip features từ Dual-Encoder.
    
    Cấu hình kênh thực tế:
        Bottleneck (BN): s5(2048) + d5(1024) = 3072 ch
        Level 4 (dec4): skip(s4:1024 + d4:1024 = 2048) + prev(1024)
        Level 3 (dec3): skip(s3:512  + d3:512  = 1024) + prev(512)
        Level 2 (dec2): skip(s2:256  + d2:256  = 512)  + prev(256)
        Level 1 (dec1): skip(s1:64   + d1:64   = 128)  + prev(128)
    """

    def __init__(self, config: dict):
        super().__init__()
        
        dec_ch = config["decoder"]["out_channels"] # [512, 256, 128, 64]
        final_ch = config["decoder"].get("final_ch", 16)
        attn_type = config["decoder"].get("attention_type", "dual")

        # Bottleneck: Nén thông tin từ mức sâu nhất
        self.bottleneck = nn.Sequential(
            ConvBnGelu(3072, 1024),
            ConvBnGelu(1024, 1024),
        )

        # 4 Cấp giải mã với Skip Connections
        self.dec4 = DecoderBlock(1024, 2048, dec_ch[0], attn_type)
        self.dec3 = DecoderBlock(dec_ch[0], 1024, dec_ch[1], attn_type)
        self.dec2 = DecoderBlock(dec_ch[1], 512, dec_ch[2], attn_type)
        self.dec1 = DecoderBlock(dec_ch[2], 128, dec_ch[3], attn_type)

        # Upsample cuối cùng và tinh chỉnh đặc trưng
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnGelu(dec_ch[3], final_ch * 2),
            ConvBnGelu(final_ch * 2, final_ch),
        )

    def forward(
        self,
        cta_skips: List[torch.Tensor],
        perf_skips: List[torch.Tensor],
    ) -> torch.Tensor:
        # Lấy đặc trưng từ Encoders (ResNet & DenseNet)
        s1, s2, s3, s4, s5 = cta_skips
        d1, d2, d3, d4, d5 = perf_skips

        # Bước 1: Bottleneck
        x = torch.cat([s5, d5], dim=1) # (B, 3072, H/32, W/32)
        x = self.bottleneck(x)         # (B, 1024, H/32, W/32)

        # Bước 2: Giải mã qua các tầng có Attention
        x = self.dec4(x, s4, d4)       # (B, 512, H/16, W/16)
        x = self.dec3(x, s3, d3)       # (B, 256, H/8, W/8)
        x = self.dec2(x, s2, d2)       # (B, 128, H/4, W/4)
        x = self.dec1(x, s1, d1)       # (B, 64, H/2, W/2)

        # Bước 3: Upsample về kích thước gốc
        x = self.final_upsample(x)     # (B, final_ch, H, W)
        return x
