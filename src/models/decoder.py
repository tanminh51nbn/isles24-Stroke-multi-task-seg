"""
decoder.py — UNet Decoder "Decoupled Specialist" v4 (LVO Sát Thủ)

Thiết kế v4:
    1. Tách đôi Decoder hoàn toàn từ tầng dec4.
    2. LVO Path: Lộ trình giải mã dành riêng cho điểm tắc (Heatmap).
    3. Lesion & Anatomy Path: Lộ trình giải mã cho vùng tổn thương và mạch máu (Guidance).
    4. Tránh hiện tượng "feature poisoning" (nhiễu đặc trưng) giữa vùng lớn và điểm nhỏ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ─── Attention Modules ───────────────────────────────────────────────────────

class AttentionGate(nn.Module):
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
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_weight = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_weight
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        spatial_mask = torch.cat([avg_mask, max_mask], dim=1)
        spatial_weight = self.sigmoid(self.spatial_conv(spatial_mask))
        return identity + (x * spatial_weight)


# ─── Khối Decoder Cơ Bản ─────────────────────────────────────────────────────

class ConvBnGelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)


class ConvBnGelu1x1(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)


class AuxHead(nn.Module):
    def __init__(self, in_ch: int, task_name: str, out_ch: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        with torch.no_grad():
            bias_val = -4.595 if task_name in ["lvo", "cow", "shared"] else -2.944
            for i in range(out_ch):
                self.conv.bias[i] = bias_val

    def forward(self, x): return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention_type: Optional[str] = "dual", use_aux: bool = True, task_name: str = "shared", aux_ch: int = 1):
        super().__init__()
        self.attention_type = attention_type
        self.use_aux = use_aux
        self.aux_ch = aux_ch
        
        self.conv1 = ConvBnGelu1x1(in_ch + skip_ch + aux_ch, out_ch)
        self.conv2 = ConvBnGelu(out_ch, out_ch)
        self.dropout = nn.Dropout2d(p=0.2)
        
        if attention_type == "ag":
            self.ag = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=skip_ch // 2)
        elif attention_type == "dual":
            self.dual_attn = LightweightDualAttention(channels=skip_ch)
            
        if use_aux:
            self.aux_head = AuxHead(out_ch, task_name, out_ch=aux_ch)

    def forward(self, x: torch.Tensor, skip_cta: torch.Tensor, skip_perf: torch.Tensor, prev_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if prev_mask is None:
            prev_mask = torch.zeros((x_up.shape[0], self.aux_ch, x_up.shape[2], x_up.shape[3]), device=x.device)
        else:
            prev_mask = F.interpolate(prev_mask, size=(x_up.shape[2], x_up.shape[3]), mode="bilinear", align_corners=False)
        
        skip = torch.cat([skip_cta, skip_perf], dim=1)
        if self.attention_type == "ag":
            skip = self.ag(g=x_up, x=skip)
        elif self.attention_type == "dual":
            skip = self.dual_attn(skip)
            
        out = torch.cat([x_up, skip, prev_mask], dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.dropout(out)
        aux_out = self.aux_head(out) if self.use_aux else None
        return out, aux_out


# ─── Specialized Decoder Paths ──────────────────────────────────────────────

class DecoupledPath(nn.Module):
    """
    Một nhánh Decoder chuyên biệt đi từ level 4 đến level 1.
    """
    def __init__(self, config: dict, task_name: str, skip_channels: List[int], aux_ch: int = 1):
        super().__init__()
        dec_ch = config["decoder"]["out_channels"]
        final_ch = config["decoder"].get("final_ch", 16)
        attn_type = config["decoder"].get("attention_type", "dual")

        # skip_channels: [s4, s3, s2, s1] -> [2048, 1024, 512, 128]
        self.dec4 = DecoderBlock(1024, skip_channels[0], dec_ch[0], attn_type, use_aux=True, task_name=task_name, aux_ch=aux_ch)
        self.dec3 = DecoderBlock(dec_ch[0], skip_channels[1], dec_ch[1], attn_type, use_aux=True, task_name=task_name, aux_ch=aux_ch)
        self.dec2 = DecoderBlock(dec_ch[1], skip_channels[2], dec_ch[2], attn_type, use_aux=True, task_name=task_name, aux_ch=aux_ch)
        self.dec1 = DecoderBlock(dec_ch[2], skip_channels[3], dec_ch[3], attn_type, use_aux=True, task_name=task_name, aux_ch=aux_ch)

        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = ConvBnGelu(dec_ch[3], final_ch)

    def forward(self, x_bottleneck, cta_skips, perf_skips):
        # skips: s1..s5
        s1, s2, s3, s4, _ = cta_skips
        d1, d2, d3, d4, _ = perf_skips

        x, aux4 = self.dec4(x_bottleneck, s4, d4)
        x, aux3 = self.dec3(x, s3, d3, prev_mask=aux4)
        x, aux2 = self.dec2(x, s2, d2, prev_mask=aux3)
        x, aux1 = self.dec1(x, s1, d1, prev_mask=aux2)

        x = self.up_final(x)
        x = self.final_conv(x)
        
        return x, [aux4, aux3, aux2, aux1]


class MultiHeadDecoder(nn.Module):
    """
    Multi-Head Decoder (LVO Sát Thủ v4):
    - Tách biệt hoàn toàn hai nhánh xử lý từ sau Bottleneck.
    - Nhánh 1 (LVO_Path): Chuyên gia soi điểm nhỏ (Heatmap).
    - Nhánh 2 (LesionAnatomy_Path): Chuyên gia phân vùng lớn (Lesion + CoW).
    """
    def __init__(self, config: dict):
        super().__init__()
        bottleneck_ch = 1024
        
        # 1. Bottleneck chung (32x32)
        self.shared_bottleneck = nn.Sequential(
            ConvBnGelu1x1(3072, bottleneck_ch),
            ConvBnGelu(bottleneck_ch, bottleneck_ch),
        )

        # 2. Hai lộ trình tách biệt
        # Skip channels (combined CTA + Perf): s4=2048, s3=1024, s2=512, s1=128
        skips = [2048, 1024, 512, 128]
        
        self.lvo_path = DecoupledPath(config, "lvo", skips, aux_ch=1)
        self.lesion_path = DecoupledPath(config, "lesion", skips, aux_ch=2) # 2 kênh: [Lesion, CoW]

    def forward(self, cta_skips: List[torch.Tensor], perf_skips: List[torch.Tensor]):
        s5, d5 = cta_skips[4], perf_skips[4]

        # 1. Shared Bottleneck
        x_bottleneck = torch.cat([s5, d5], dim=1)
        x_bottleneck = self.shared_bottleneck(x_bottleneck)
        
        # 2. LVO Path (Sát thủ điểm tắc)
        f_lvo, lvo_auxs = self.lvo_path(x_bottleneck, cta_skips, perf_skips)

        # 3. Lesion & Anatomy Path (Chuyên gia vùng lớn)
        f_lesion_cow, lesion_auxs = self.lesion_path(x_bottleneck, cta_skips, perf_skips)

        # Trích xuất f_lesion và f_cow từ nhánh 2
        # (Ở đây ta dùng 1 head chung cho Lesion và CoW để tiết kiệm tài nguyên, 
        #  hoặc trả về feature map để head.py tách ra sau)
        # Để thống nhất với MultiTaskHeads cũ, ta trả về feature map shared cho Lesion/CoW
        
        # Gom nhóm Aux Masks: [32, 64, 128, 256]
        # Mỗi tầng chứa: [Lesion, LVO, CoW]
        # lesion_auxs[i] có 2 kênh: [0: Lesion, 1: CoW]
        # lvo_auxs[i] có 1 kênh: [0: LVO]
        aux_masks = []
        for i in range(4):
            # Cấu trúc kênh: [Lesion, LVO, CoW]
            aux = torch.cat([
                lesion_auxs[i][:, 0:1], # Lesion
                lvo_auxs[i],           # LVO
                lesion_auxs[i][:, 1:2]  # CoW
            ], dim=1)
            aux_masks.append(aux)

        preds = {
            "lesion": f_lesion_cow,
            "lvo":    f_lvo,
            "cow":    f_lesion_cow # Dùng chung feature map, head sẽ tự tách
        }
        
        return preds, aux_masks, {} # Không dùng g_maps cũ
