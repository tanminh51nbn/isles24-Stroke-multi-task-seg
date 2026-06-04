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
            # [FIX] Đổi CoW bias từ -4.595 thành -2.944. -4.595 quá nhỏ khiến prediction ~0, gradient = 0
            bias_val = -4.595 if task_name in ["lvo", "shared"] else -2.944
            for i in range(out_ch):
                self.conv.bias[i] = bias_val

    def forward(self, x): return self.conv(x)


class FeatureFusionBlock(nn.Module):
    """
    [NEW] Đồng bộ hóa ngữ nghĩa (Semantic Synchronization):
    Đưa đặc trưng từ 2 Encoder về cùng một không gian biểu diễn và dùng 
    Channel-Attention để mô hình tự học trọng số giữa CTA (Cấu trúc) và Perfusion (Tưới máu).
    """
    def __init__(self, cta_ch: int, perf_ch: int):
        super().__init__()
        self.conv_cta = ConvBnGelu1x1(cta_ch, cta_ch)
        self.conv_perf = ConvBnGelu1x1(perf_ch, perf_ch)
        
        in_ch = cta_ch + perf_ch
        # Squeeze-and-Excitation để học trọng số kênh
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_ch // 8, in_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, cta, perf):
        cta = self.conv_cta(cta)
        perf = self.conv_perf(perf)
        fused = torch.cat([cta, perf], dim=1)
        return fused * self.se(fused)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, cta_skip_ch: int, perf_skip_ch: int, out_ch: int, attention_type: Optional[str] = "dual", use_aux: bool = True, task_name: str = "shared", aux_ch: int = 1):
        super().__init__()
        self.attention_type = attention_type
        self.use_aux = use_aux
        self.aux_ch = aux_ch
        self.task_name = task_name
        
        skip_ch = cta_skip_ch + perf_skip_ch
        # Module Fusion để đồng bộ CTA và Perfusion
        self.fusion = FeatureFusionBlock(cta_skip_ch, perf_skip_ch)
        
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
        interp_mode = "nearest" if self.task_name == "lvo" else "bilinear"
        
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if prev_mask is None:
            prev_mask = torch.zeros((x_up.shape[0], self.aux_ch, x_up.shape[2], x_up.shape[3]), device=x.device)
        else:
            prev_mask = F.interpolate(prev_mask, size=(x_up.shape[2], x_up.shape[3]), mode=interp_mode, align_corners=(False if interp_mode == "bilinear" else None))
        
        skip = self.fusion(skip_cta, skip_perf)
        
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

class SharedPath(nn.Module):
    """
    Nhánh giải mã dùng chung (Shared) cho các tầng sâu (16x16, 32x32).
    Ép mô hình học chung bối cảnh giải phẫu tổng thể.
    """
    def __init__(self, config: dict, cta_skip_channels: List[int], perf_skip_channels: List[int]):
        super().__init__()
        dec_ch = config["decoder"]["out_channels"]
        attn_type = config["decoder"].get("attention_type", "dual")

        self.dec4 = DecoderBlock(1024, cta_skip_channels[0], perf_skip_channels[0], dec_ch[0], attn_type, use_aux=False, aux_ch=0)
        self.dec3 = DecoderBlock(dec_ch[0], cta_skip_channels[1], perf_skip_channels[1], dec_ch[1], attn_type, use_aux=False, aux_ch=0)

    def forward(self, x_bottleneck, cta_skips_shared, perf_skips_shared):
        s4, s3 = cta_skips_shared
        d4, d3 = perf_skips_shared
        x, _ = self.dec4(x_bottleneck, s4, d4, prev_mask=None)
        x, _ = self.dec3(x, s3, d3, prev_mask=None)
        return x

class TaskPath(nn.Module):
    """
    Một nhánh Decoder chuyên biệt đi từ level 2 đến level 1.
    """
    def __init__(self, in_ch: int, config: dict, task_name: str, cta_skip_channels: List[int], perf_skip_channels: List[int], aux_ch: int = 1, active_aux_levels: List[bool] = [True, True], guidance_ch: int = 0):
        super().__init__()
        self.task_name = task_name
        dec_ch = config["decoder"]["out_channels"]
        final_ch = config["decoder"].get("final_ch", 16)
        attn_type = config["decoder"].get("attention_type", "dual")

        self.dec2 = DecoderBlock(in_ch, cta_skip_channels[0], perf_skip_channels[0], dec_ch[2], attn_type, use_aux=active_aux_levels[0], task_name=task_name, aux_ch=aux_ch)
        self.dec1 = DecoderBlock(dec_ch[2], cta_skip_channels[1], perf_skip_channels[1], dec_ch[3], attn_type, use_aux=active_aux_levels[1], task_name=task_name, aux_ch=aux_ch)

        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = ConvBnGelu(dec_ch[3], final_ch)
        
        if guidance_ch > 0:
            self.guidance_fusion = nn.Sequential(
                nn.Conv2d(final_ch + guidance_ch, final_ch, kernel_size=1),
                nn.BatchNorm2d(final_ch),
                nn.GELU()
            )
        else:
            self.guidance_fusion = None

    def forward(self, x_shared, cta_skips_task, perf_skips_task, guidance: Optional[torch.Tensor] = None):
        s2, s1 = cta_skips_task
        d2, d1 = perf_skips_task

        x, aux2 = self.dec2(x_shared, s2, d2, prev_mask=None)
        x, aux1 = self.dec1(x, s1, d1, prev_mask=aux2)

        x = self.up_final(x)
        x = self.final_conv(x)
        
        if guidance is not None and self.guidance_fusion is not None:
            g_interp = F.interpolate(guidance, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = self.guidance_fusion(torch.cat([x, g_interp], dim=1))

        return x, [None, None, aux2, aux1]


class MultiHeadDecoder(nn.Module):
    """
    Multi-Head Decoder (Partially Shared v5):
    - Khối SharedPath: Dùng chung ở độ phân giải thấp (bối cảnh giải phẫu).
    - Khối TaskPath: Tách nhánh chuyên gia ở độ phân giải cao (Cow, Lesion, LVO).
    """
    def __init__(self, config: dict, cta_channels: List[int], perf_channels: List[int]):
        super().__init__()
        bottleneck_ch = 1024
        
        # 1. Bottleneck chung (32x32)
        in_bottleneck = cta_channels[4] + perf_channels[4]
        self.shared_bottleneck = nn.Sequential(
            ConvBnGelu1x1(in_bottleneck, bottleneck_ch),
            ConvBnGelu(bottleneck_ch, bottleneck_ch),
        )

        cta_skips_shared = [cta_channels[3], cta_channels[2]]  # s4, s3
        perf_skips_shared = [perf_channels[3], perf_channels[2]] # d4, d3
        
        cta_skips_task = [cta_channels[1], cta_channels[0]]    # s2, s1
        perf_skips_task = [perf_channels[1], perf_channels[0]]   # d2, d1
        
        self.shared_path = SharedPath(config, cta_skips_shared, perf_skips_shared)
        
        dec_ch = config["decoder"]["out_channels"]
        in_ch_task = dec_ch[1] # out of dec3
        
        self.cow_path    = TaskPath(in_ch_task, config, "cow", cta_skips_task, perf_skips_task, aux_ch=1, active_aux_levels=[True, True], guidance_ch=0)
        self.lvo_path    = TaskPath(in_ch_task, config, "lvo", cta_skips_task, perf_skips_task, aux_ch=1, active_aux_levels=[False, False], guidance_ch=16)
        self.lesion_path = TaskPath(in_ch_task, config, "lesion", cta_skips_task, perf_skips_task, aux_ch=1, active_aux_levels=[True, True], guidance_ch=32)

    def forward(self, cta_skips: List[torch.Tensor], perf_skips: List[torch.Tensor], epoch: int = 0):
        s1, s2, s3, s4, s5 = cta_skips
        d1, d2, d3, d4, d5 = perf_skips

        # 1. Shared Bottleneck
        x_bottleneck = torch.cat([s5, d5], dim=1)
        x_bottleneck = self.shared_bottleneck(x_bottleneck)
        
        # 2. Shared Path (dec4, dec3)
        x_shared = self.shared_path(x_bottleneck, [s4, s3], [d4, d3])
        
        # 3. DÒNG CHẢY TRI THỨC (Knowledge Cascade): CoW -> LVO -> Lesion
        f_cow, cow_auxs = self.cow_path(x_shared, [s2, s1], [d2, d1])
        
        f_lvo, lvo_auxs = self.lvo_path(x_shared, [s2, s1], [d2, d1], guidance=f_cow.detach())
        
        guidance_for_lesion = torch.cat([f_cow.detach(), f_lvo.detach()], dim=1)
        f_lesion, lesion_auxs = self.lesion_path(x_shared, [s2, s1], [d2, d1], guidance=guidance_for_lesion)

        aux_masks = {
            "lesion": lesion_auxs,
            "lvo":    lvo_auxs,
            "cow":    cow_auxs
        }

        preds = {
            "lesion": f_lesion,
            "lvo":    f_lvo,
            "cow":    f_cow
        }
        
        return preds, aux_masks, {}
