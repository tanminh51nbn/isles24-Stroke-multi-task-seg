"""
decoder.py — UNet Decoder tích hợp Iterative Feedback MDS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ─── Attention Modules ───────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) chuẩn.
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
    CBAM-style Dual Attention với KẾT NỐI RESIDUAL (Bảo hiểm tín hiệu).
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        b, c, _, _ = x.size()
        
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_weight = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_weight

        # Spatial Attention
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        spatial_mask = torch.cat([avg_mask, max_mask], dim=1)
        spatial_weight = self.sigmoid(self.spatial_conv(spatial_mask))
        
        return identity + (x * spatial_weight)


# ─── Khối Decoder Cơ Bản ─────────────────────────────────────────────────────

class ConvBnGelu(nn.Module):
    """3×3 Conv + BN + GELU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class AuxHead(nn.Module):
    """Tạo mask 1 kênh từ đặc trưng trung gian của nhánh tương ứng."""
    def __init__(self, in_ch: int, task_name: str):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        
        # [QUAN TRỌNG] Bias Initialization tương tự như Main Head
        # Tránh Aux Loss bùng nổ (Lên tới hàng trăm) ở Epoch 1
        with torch.no_grad():
            if task_name == "lesion":
                self.conv.bias[0] = -2.944
            elif task_name in ["lvo", "cow"]:
                self.conv.bias[0] = -4.595

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """
    DecoderBlock hỗ trợ Iterative Feedback (1 kênh cho task cụ thể).
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention_type: Optional[str] = None, use_aux: bool = True, task_name: str = "lesion"):
        super().__init__()
        self.attention_type = attention_type
        self.use_aux = use_aux
        
        # Feedback (1 channel) + In_ch + Skip_ch
        self.conv1 = ConvBnGelu(in_ch + skip_ch + 1, out_ch)
        self.conv2 = ConvBnGelu(out_ch, out_ch)
        
        if attention_type == "ag":
            self.ag = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=skip_ch // 2)
        elif attention_type == "dual":
            self.dual_attn = LightweightDualAttention(channels=skip_ch)
            
        if use_aux:
            self.aux_head = AuxHead(out_ch, task_name)

    def forward(self, x: torch.Tensor, skip_cta: torch.Tensor, skip_perf: torch.Tensor, prev_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Upsample
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        
        if prev_mask is None:
            prev_mask = torch.zeros((x_up.shape[0], 1, x_up.shape[2], x_up.shape[3]), device=x.device)
        else:
            prev_mask = F.interpolate(prev_mask, size=(x_up.shape[2], x_up.shape[3]), mode="bilinear", align_corners=False)
        
        # 2. Skip + Attention
        skip = torch.cat([skip_cta, skip_perf], dim=1)
        if self.attention_type == "ag":
            skip = self.ag(g=x_up, x=skip)
        elif self.attention_type == "dual":
            skip = self.dual_attn(skip)
            
        # 3. Concat & Conv
        out = torch.cat([x_up, skip, prev_mask], dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        
        aux_out = self.aux_head(out) if self.use_aux else None
        return out, aux_out


# ─── UNet Decoder ───────────────────────────────────────────────────────────

class TaskDecoder(nn.Module):
    """
    Một nhánh Decoder độc lập cho một Task cụ thể.
    Chỉ bắt đầu từ tầng sau Bottleneck để tiết kiệm VRAM.
    """
    def __init__(self, config: dict, task_name: str):
        super().__init__()
        dec_ch = config["decoder"]["out_channels"] # [512, 256, 128, 64]
        final_ch = config["decoder"].get("final_ch", 16)
        attn_type = config["decoder"].get("attention_type", "dual")

        # 4 Cấp giải mã
        self.dec4 = DecoderBlock(1024, 2048, dec_ch[0], attn_type, use_aux=True, task_name=task_name)
        self.dec3 = DecoderBlock(dec_ch[0], 1024, dec_ch[1], attn_type, use_aux=True, task_name=task_name)
        self.dec2 = DecoderBlock(dec_ch[1], 512, dec_ch[2], attn_type, use_aux=True, task_name=task_name)
        self.dec1 = DecoderBlock(dec_ch[2], 128, dec_ch[3], attn_type, use_aux=True, task_name=task_name)

        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = ConvBnGelu(dec_ch[3], final_ch)

    def forward(self, x, cta_skips, perf_skips):
        s1, s2, s3, s4, _ = cta_skips
        d1, d2, d3, d4, _ = perf_skips

        x, aux4 = self.dec4(x, s4, d4, prev_mask=None)       # 32x32
        x, aux3 = self.dec3(x, s3, d3, prev_mask=aux4)       # 64x64
        x, aux2 = self.dec2(x, s2, d2, prev_mask=aux3)       # 128x128
        x, aux1 = self.dec1(x, s1, d1, prev_mask=aux2)       # 256x256

        x = self.up_final(x)
        x = self.final_conv(x)
        
        # Trả về: (đặc trưng cuối 256x256, [mask_32, mask_64, mask_128, mask_256_aux])
        # Chúng ta có 4 aux masks bây giờ
        return x, [aux4, aux3, aux2, aux1]

class MultiHeadDecoder(nn.Module):
    """
    Multi-Head Decoder: 
    Nhận skips từ 2 encoder, nén qua Bottleneck chung, sau đó rẽ làm 3 nhánh TaskDecoder riêng.
    """
    def __init__(self, config: dict):
        super().__init__()
        
        # Bottleneck chung: Nén s5(2048) + d5(1024) = 3072 ch về 1024
        self.bottleneck = nn.Sequential(
            ConvBnGelu(3072, 1024),
            ConvBnGelu(1024, 1024),
        )

        # 3 Nhánh độc lập
        self.lesion_dec = TaskDecoder(config, "lesion")
        self.lvo_dec    = TaskDecoder(config, "lvo")
        self.cow_dec    = TaskDecoder(config, "cow")

    def forward(self, cta_skips: List[torch.Tensor], perf_skips: List[torch.Tensor]):
        # Lấy s5, d5 cho bottleneck
        s5 = cta_skips[4]
        d5 = perf_skips[4]

        # Shared Bottleneck
        x = torch.cat([s5, d5], dim=1)
        x = self.bottleneck(x)

        # 3 Nhánh thực thi độc lập (Chống xung đột)
        f_lesion, aux_lesion = self.lesion_dec(x, cta_skips, perf_skips)
        f_lvo,    aux_lvo    = self.lvo_dec(x, cta_skips, perf_skips)
        f_cow,    aux_cow    = self.cow_dec(x, cta_skips, perf_skips)

        # Gộp các Aux Masks lại thành Tensor (B, 3, H, W) để tương thích chuẩn với losses.py
        aux_masks = []
        for i in range(4):
            if aux_lesion[i] is not None:
                mask = torch.cat([aux_lesion[i], aux_lvo[i], aux_cow[i]], dim=1)
                aux_masks.append(mask)
            else:
                aux_masks.append(None)

        features = {
            "lesion": f_lesion,
            "lvo":    f_lvo,
            "cow":    f_cow
        }
        
        return features, aux_masks
