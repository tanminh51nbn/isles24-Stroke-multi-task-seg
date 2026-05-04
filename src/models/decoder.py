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


class ConvBnGelu1x1(nn.Module):
    """1×1 Conv + BN + GELU (Bottleneck)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class AuxHead(nn.Module):
    """Tạo mask (1 hoặc 3 kênh) từ đặc trưng trung gian."""
    def __init__(self, in_ch: int, task_name: str, out_ch: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        
        with torch.no_grad():
            if task_name == "lesion":
                self.conv.bias[0] = -2.944
            elif task_name in ["lvo", "cow", "shared", "ischemic"]:
                self.conv.bias[0] = -4.595
                if out_ch > 1: # Cho shared dec4 hoặc ischemic
                    for i in range(1, out_ch):
                        self.conv.bias[i] = -4.595

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """
    DecoderBlock Bottleneck: 1x1 Conv nén kênh + 3x3 Conv học không gian.
    Giúp giảm tham số cực lớn khi skip_ch cao (vd: 2048).
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention_type: Optional[str] = None, use_aux: bool = True, task_name: str = "lesion", aux_ch: int = 1):
        super().__init__()
        self.attention_type = attention_type
        self.use_aux = use_aux
        self.aux_ch = aux_ch
        
        # BOTTLENECK: Dùng 1x1 nén trước khi dùng 3x3
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
        # 1. Upsample
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        
        if prev_mask is None:
            prev_mask = torch.zeros((x_up.shape[0], self.aux_ch, x_up.shape[2], x_up.shape[3]), device=x.device)
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
        out = self.dropout(out)
        
        aux_out = self.aux_head(out) if self.use_aux else None
        return out, aux_out


# ─── UNet Decoder ───────────────────────────────────────────────────────────

class TaskBranch(nn.Module):
    """
    Một nhánh Decoder rẽ sau tầng dec4 dành cho các task đơn lẻ (như CoW).
    """
    def __init__(self, config: dict, task_name: str):
        super().__init__()
        dec_ch = config["decoder"]["out_channels"]
        final_ch = config["decoder"].get("final_ch", 16)
        attn_type = config["decoder"].get("attention_type", "dual")

        self.dec3 = DecoderBlock(dec_ch[0], 1024, dec_ch[1], attn_type, use_aux=True, task_name=task_name)
        self.dec2 = DecoderBlock(dec_ch[1], 512, dec_ch[2], attn_type, use_aux=True, task_name=task_name)
        self.dec1 = DecoderBlock(dec_ch[2], 128, dec_ch[3], attn_type, use_aux=True, task_name=task_name)

        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = ConvBnGelu(dec_ch[3], final_ch)

    def forward(self, x, cta_skips, perf_skips, aux4_branch):
        s1, s2, s3, _, _ = cta_skips
        d1, d2, d3, _, _ = perf_skips

        x, aux3 = self.dec3(x, s3, d3, prev_mask=aux4_branch)
        x, aux2 = self.dec2(x, s2, d2, prev_mask=aux3)
        x, aux1 = self.dec1(x, s1, d1, prev_mask=aux2)

        x = self.up_final(x)
        x = self.final_conv(x)
        
        return x, [aux3, aux2, aux1]


class LVOBranch(nn.Module):
    """
    Nhánh chuyên biệt để dự đoán LVO.
    Sử dụng Vascular Guidance để giới hạn vùng tìm kiếm trên mạch máu.
    """
    def __init__(self, config: dict):
        super().__init__()
        dec_ch = config["decoder"]["out_channels"]
        final_ch = config["decoder"].get("final_ch", 16)
        attn_type = config["decoder"].get("attention_type", "dual")

        self.dec3 = DecoderBlock(dec_ch[0], 1024, dec_ch[1], attn_type, use_aux=True, task_name="lvo")
        self.dec2 = DecoderBlock(dec_ch[1], 512, dec_ch[2], attn_type, use_aux=True, task_name="lvo")
        self.dec1 = DecoderBlock(dec_ch[2], 128, dec_ch[3], attn_type, use_aux=True, task_name="lvo")

        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        self.vascular_to_lvo_gate = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, kernel_size=1),
            nn.BatchNorm2d(final_ch // 2),
            nn.GELU(),
            nn.Conv2d(final_ch // 2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # Sử dụng phép nhân nên số kênh vào vẫn là dec_ch[3]
        self.final_conv = ConvBnGelu(dec_ch[3], final_ch)

    def forward(self, x, cta_skips, perf_skips, aux4_shared, vascular_feat):
        s1, s2, s3, _, _ = cta_skips
        d1, d2, d3, _, _ = perf_skips

        x, aux3 = self.dec3(x, s3, d3, prev_mask=aux4_shared[:, 1:2]) # LVO index
        x, aux2 = self.dec2(x, s2, d2, prev_mask=aux3)
        x, aux1 = self.dec1(x, s1, d1, prev_mask=aux2)

        x_up = self.up_final(x)
        
        # [CẢI TIẾN] Spatial Gating: Nhân trực tiếp đặc trưng với bản đồ mạch máu
        v_guidance = self.vascular_to_lvo_gate(vascular_feat)
        x_guided = x_up * (1.0 + v_guidance) 
        
        f_lvo = self.final_conv(x_guided)
        return f_lvo, [aux3, aux2, aux1], v_guidance


class LesionBranch(nn.Module):
    """
    Nhánh chuyên biệt để dự đoán Lesion.
    Sử dụng LVO Guidance để tập trung vào vùng nhồi máu quanh điểm tắc.
    """
    def __init__(self, config: dict):
        super().__init__()
        dec_ch = config["decoder"]["out_channels"]
        final_ch = config["decoder"].get("final_ch", 16)
        attn_type = config["decoder"].get("attention_type", "dual")

        self.dec3 = DecoderBlock(dec_ch[0], 1024, dec_ch[1], attn_type, use_aux=True, task_name="lesion")
        self.dec2 = DecoderBlock(dec_ch[1], 512, dec_ch[2], attn_type, use_aux=True, task_name="lesion")
        self.dec1 = DecoderBlock(dec_ch[2], 128, dec_ch[3], attn_type, use_aux=True, task_name="lesion")

        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        self.lvo_to_lesion_gate = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(final_ch // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.final_conv = ConvBnGelu(dec_ch[3], final_ch)

    def forward(self, x, cta_skips, perf_skips, aux4_shared, lvo_feat):
        s1, s2, s3, _, _ = cta_skips
        d1, d2, d3, _, _ = perf_skips

        x, aux3 = self.dec3(x, s3, d3, prev_mask=aux4_shared[:, 0:1]) # Lesion index
        x, aux2 = self.dec2(x, s2, d2, prev_mask=aux3)
        x, aux1 = self.dec1(x, s1, d1, prev_mask=aux2)

        x_up = self.up_final(x)
        
        # [CẢI TIẾN] Spatial Gating: Tập trung vào vùng nhồi máu quanh điểm tắc
        l_guidance = self.lvo_to_lesion_gate(lvo_feat)
        x_guided = x_up * (1.0 + l_guidance)
        
        f_lesion = self.final_conv(x_guided)
        return f_lesion, [aux3, aux2, aux1], l_guidance


class MultiHeadDecoder(nn.Module):
    """
    Multi-Head Decoder (Ischemic + Vascular Split):
    - Shared Bottleneck & Shared dec4 (32x32).
    - Ischemic Branch: Gộp Lesion + LVO (Tiết kiệm VRAM & Tăng cường bổ trợ).
    - Vascular Branch: CoW riêng biệt.
    """
    def __init__(self, config: dict):
        super().__init__()
        dec_ch = config["decoder"]["out_channels"]
        bottleneck_ch = 1024
        attn_type = config["decoder"].get("attention_type", "dual")
        
        # 1. Bottleneck chung (3072 -> 1024)
        self.shared_bottleneck = nn.Sequential(
            ConvBnGelu1x1(3072, bottleneck_ch),
            ConvBnGelu(bottleneck_ch, bottleneck_ch),
        )

        # 0. Shared Base (Bottleneck + dec4)
        # Tại level 4: skip_cta (1024) + skip_perf (1024) = 2048 channels
        self.dec4_shared = DecoderBlock(bottleneck_ch, 2048, dec_ch[0], attn_type, use_aux=True, task_name="shared", aux_ch=3)
        
        # 1. Nhánh Mạch máu (CoW)
        self.vascular_branch = TaskBranch(config, "cow")

        # 2. Nhánh Điểm tắc (LVO) - TÁCH RIÊNG
        self.lvo_branch = LVOBranch(config)

        # 3. Nhánh Tổn thương (Lesion) - TÁCH RIÊNG
        self.lesion_branch = LesionBranch(config)

    def forward(self, cta_skips: List[torch.Tensor], perf_skips: List[torch.Tensor]):
        s5, s4 = cta_skips[4], cta_skips[3]
        d5, d4 = perf_skips[4], perf_skips[3]

        # 1. Shared Bottleneck (32x32)
        x_bottleneck = torch.cat([s5, d5], dim=1)
        x_bottleneck = self.shared_bottleneck(x_bottleneck)
        
        # 2. Shared dec4
        x_shared, aux4_shared = self.dec4_shared(x_bottleneck, s4, d4)

        # 3. Nhánh Vascular (CoW) -> Chạy trước để làm guidance
        f_cow, cow_auxs = self.vascular_branch(x_shared, cta_skips, perf_skips, aux4_shared[:, 2:3])

        # 4. Nhánh LVO -> Chạy sau, nhận guidance từ Vascular (f_cow)
        f_lvo, lvo_auxs, v_guidance = self.lvo_branch(x_shared, cta_skips, perf_skips, aux4_shared, f_cow)

        # 5. Nhánh Lesion -> Chạy cuối, nhận guidance từ LVO (f_lvo)
        f_lesion, lesion_auxs, l_guidance = self.lesion_branch(x_shared, cta_skips, perf_skips, aux4_shared, f_lvo)

        # Trình tự Aux Masks: [32, 64, 128, 256]
        # Mỗi mask chứa [Lesion, LVO, CoW]
        aux_masks = [aux4_shared]
        for i in range(3): # aux3, aux2, aux1
            mask = torch.cat([lesion_auxs[i], lvo_auxs[i], cow_auxs[i]], dim=1)
            aux_masks.append(mask)

        # Guidance maps để debug
        g_maps = {
            "v_guidance": v_guidance,
            "l_guidance": l_guidance
        }

        preds = {"lesion": f_lesion, "lvo": f_lvo, "cow": f_cow}
        return preds, aux_masks, g_maps
