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


class FeatureFusionBlock(nn.Module):
    """
    [NEW] Đồng bộ hóa ngữ nghĩa (Semantic Synchronization):
    Đưa đặc trưng từ 2 Encoder về cùng một không gian biểu diễn và dùng 
    Channel-Attention để mô hình tự học trọng số giữa CTA (Cấu trúc) và Perfusion (Tưới máu).
    """
    def __init__(self, in_ch: int):
        super().__init__()
        # Giả định cta_ch == perf_ch == in_ch // 2
        half_ch = in_ch // 2
        self.conv_cta = ConvBnGelu1x1(half_ch, half_ch)
        self.conv_perf = ConvBnGelu1x1(half_ch, half_ch)
        
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
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention_type: Optional[str] = "dual", use_aux: bool = True, task_name: str = "shared", aux_ch: int = 1):
        super().__init__()
        self.attention_type = attention_type
        self.use_aux = use_aux
        self.aux_ch = aux_ch
        self.task_name = task_name
        
        # Module Fusion để đồng bộ CTA và Perfusion
        self.fusion = FeatureFusionBlock(skip_ch)
        
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

class DecoupledPath(nn.Module):
    """
    Một nhánh Decoder chuyên biệt đi từ level 4 đến level 1.
    """
    def __init__(self, config: dict, task_name: str, skip_channels: List[int], aux_ch: int = 1, active_aux_levels: List[bool] = [True, True, True, True]):
        super().__init__()
        self.active_aux_levels = active_aux_levels
        self.task_name = task_name
        dec_ch = config["decoder"]["out_channels"]
        final_ch = config["decoder"].get("final_ch", 16)
        attn_type = config["decoder"].get("attention_type", "dual")

        # skip_channels: [s4, s3, s2, s1] -> [2048, 1024, 512, 128]
        self.dec4 = DecoderBlock(1024, skip_channels[0], dec_ch[0], attn_type, use_aux=active_aux_levels[0], task_name=task_name, aux_ch=aux_ch)
        self.dec3 = DecoderBlock(dec_ch[0], skip_channels[1], dec_ch[1], attn_type, use_aux=active_aux_levels[1], task_name=task_name, aux_ch=aux_ch)
        self.dec2 = DecoderBlock(dec_ch[1], skip_channels[2], dec_ch[2], attn_type, use_aux=active_aux_levels[2], task_name=task_name, aux_ch=aux_ch)
        self.dec1 = DecoderBlock(dec_ch[2], skip_channels[3], dec_ch[3], attn_type, use_aux=active_aux_levels[3], task_name=task_name, aux_ch=aux_ch)

        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = ConvBnGelu(dec_ch[3], final_ch)
        
        # [NEW] Cầu nối tri thức: Tiếp nhận thông tin từ các nhánh khác
        self.guidance_conv = nn.Conv2d(final_ch, final_ch, kernel_size=1) if task_name != "cow" else None

    def forward(self, x_bottleneck, cta_skips, perf_skips, guidance: Optional[torch.Tensor] = None, epoch: int = 0):
        # skips: s1..s5
        s1, s2, s3, s4, _ = cta_skips
        d1, d2, d3, d4, _ = perf_skips

        x, aux4 = self.dec4(x_bottleneck, s4, d4)
        x, aux3 = self.dec3(x, s3, d3, prev_mask=aux4)
        x, aux2 = self.dec2(x, s2, d2, prev_mask=aux3)
        x, aux1 = self.dec1(x, s1, d1, prev_mask=aux2)

        x = self.up_final(x)
        x = self.final_conv(x)
        
        # Cộng thêm tri thức từ nhánh khác nếu có (Guidance)
        if guidance is not None and self.guidance_conv is not None:
            g = self.guidance_conv(F.interpolate(guidance, size=x.shape[2:], mode='bilinear', align_corners=False))

            if self.task_name == "lvo" and epoch >= 20:
                # [T2.2] Territory-Aware Attention: CoW map làm spatial gate cho LVO.
                # Chỉ kích hoạt từ epoch 20 (khi CoW đã ổn định ~0.9 Dice) để tránh noise.
                vessel_prob = torch.sigmoid(g)
                vessel_mask = F.max_pool2d(vessel_prob, kernel_size=15, stride=1, padding=7)
                x = x * (0.3 + 0.7 * vessel_mask)
            else:
                # Nhánh Lesion hoặc LVO (giai đoạn đầu): dùng additive guidance
                x = x + g

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

        # 2. Ba lộ trình tách biệt hoàn toàn (Triple-Head Specialist)
        # Skip channels (combined CTA + Perf): s4=2048, s3=1024, s2=512, s1=128
        skips = [2048, 1024, 512, 128]
        
        # [FIX] Cấu hình AUX: LVO tắt 2 tầng sâu (16x16, 32x32) để giảm nhiễu
        self.lesion_path = DecoupledPath(config, "lesion", skips, aux_ch=1, active_aux_levels=[True, True, True, True])
        self.lvo_path    = DecoupledPath(config, "lvo", skips, aux_ch=1, active_aux_levels=[False, False, True, True])
        self.cow_path    = DecoupledPath(config, "cow", skips, aux_ch=1, active_aux_levels=[True, True, True, True])

    def forward(self, cta_skips: List[torch.Tensor], perf_skips: List[torch.Tensor], epoch: int = 0):
        s5, d5 = cta_skips[4], perf_skips[4]

        # 1. Shared Bottleneck
        x_bottleneck = torch.cat([s5, d5], dim=1)
        x_bottleneck = self.shared_bottleneck(x_bottleneck)
        
        # 2. DÒNG CHẢY TRI THỨC (Knowledge Cascade)
        # Bước 1: Nhánh CoW chạy trước (Xây dựng bản đồ giải phẫu)
        f_cow, cow_auxs = self.cow_path(x_bottleneck, cta_skips, perf_skips, epoch=epoch)
        
        # Bước 2: Nhánh LVO học từ CoW (Tìm điểm tắc trên mạch máu)
        f_lvo, lvo_auxs = self.lvo_path(x_bottleneck, cta_skips, perf_skips, guidance=f_cow, epoch=epoch)
        
        # Bước 3: Nhánh Lesion học từ cả hai (Tìm vùng tổn thương dựa trên mạch máu và điểm tắc)
        # Gộp thông tin từ CoW và LVO để dẫn đường cho Lesion
        guidance_for_lesion = f_cow + f_lvo 
        f_lesion, lesion_auxs = self.lesion_path(x_bottleneck, cta_skips, perf_skips, guidance=guidance_for_lesion, epoch=epoch)

        # Trả về Dictionary AUX để losses.py xử lý chọn lọc
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
