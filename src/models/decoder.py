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


class IschemicBranch(nn.Module):
    """
    Nhánh Ischemic dùng chung cho Lesion và LVO.
    Tích hợp cơ chế Clinical Guidance: Vascular -> LVO -> Lesion.
    """
    def __init__(self, config: dict):
        super().__init__()
        dec_ch = config["decoder"]["out_channels"]
        final_ch = config["decoder"].get("final_ch", 16)
        attn_type = config["decoder"].get("attention_type", "dual")

        self.dec3 = DecoderBlock(dec_ch[0], 1024, dec_ch[1], attn_type, use_aux=True, task_name="ischemic", aux_ch=2)
        self.dec2 = DecoderBlock(dec_ch[1], 512, dec_ch[2], attn_type, use_aux=True, task_name="ischemic", aux_ch=2)
        self.dec1 = DecoderBlock(dec_ch[2], 128, dec_ch[3], attn_type, use_aux=True, task_name="ischemic", aux_ch=2)

        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        # ─── LVO Head với Vascular Guidance ───
        # LVO chỉ có thể xuất hiện TRÊN mạch máu (CoW)
        self.vascular_to_lvo_gate = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, kernel_size=1),
            nn.BatchNorm2d(final_ch // 2),
            nn.GELU(),

            nn.Conv2d(final_ch // 2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.final_conv_lvo = ConvBnGelu(dec_ch[3] + 1, final_ch)
        
        # ─── Lesion Head với LVO Guidance ───
        # Vùng nhồi máu thường bao quanh điểm tắc mạch
        self.lvo_to_lesion_gate = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(final_ch // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.final_conv_lesion = ConvBnGelu(dec_ch[3] + 1, final_ch)

    def forward(self, x, cta_skips, perf_skips, aux4_shared, vascular_feat):
        """
        Args:
            vascular_feat: Đặc trưng từ nhánh CoW (Vascular)
        """
        s1, s2, s3, _, _ = cta_skips
        d1, d2, d3, _, _ = perf_skips

        x, aux3 = self.dec3(x, s3, d3, prev_mask=aux4_shared[:, 0:2])
        x, aux2 = self.dec2(x, s2, d2, prev_mask=aux3)
        x, aux1 = self.dec1(x, s1, d1, prev_mask=aux2)

        x_up = self.up_final(x)
        
        # 1. Vascular Guidance -> LVO
        v_guidance = self.vascular_to_lvo_gate(vascular_feat)
        f_lvo = self.final_conv_lvo(torch.cat([x_up, v_guidance], dim=1))
        
        # 2. LVO Guidance -> Lesion
        l_guidance = self.lvo_to_lesion_gate(f_lvo)
        f_lesion = self.final_conv_lesion(torch.cat([x_up, l_guidance], dim=1))
        
        # Trả về thêm guidance maps để phục vụ Debug (Khả năng 1)
        guidance_outputs = {
            "v_guidance": v_guidance, # Vascular -> LVO
            "l_guidance": l_guidance  # LVO -> Lesion
        }
        
        return f_lesion, f_lvo, [aux3, aux2, aux1], guidance_outputs


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
        attn_type = config["decoder"].get("attention_type", "dual")
        
        # 1. Bottleneck chung (3072 -> 1024)
        self.shared_bottleneck = nn.Sequential(
            ConvBnGelu1x1(3072, 1024),
            ConvBnGelu(1024, 1024),
        )

        # 2. Shared dec4 (32x32)
        # aux_ch=3 (lesion, lvo, cow)
        self.shared_dec4 = DecoderBlock(1024, 2048, dec_ch[0], attn_type, use_aux=True, task_name="shared", aux_ch=3)

        # 3. Hai nhánh chính
        self.ischemic_branch = IschemicBranch(config)
        self.cow_branch      = TaskBranch(config, "cow")

    def forward(self, cta_skips: List[torch.Tensor], perf_skips: List[torch.Tensor]):
        s5, s4 = cta_skips[4], cta_skips[3]
        d5, d4 = perf_skips[4], perf_skips[3]

        # Shared processing
        x = torch.cat([s5, d5], dim=1)
        x = self.shared_bottleneck(x)
        x, aux4_3ch = self.shared_dec4(x, s4, d4, prev_mask=None)

        # Splitting
        # 1. Nhánh Vascular (CoW) chạy trước để tạo guidance
        f_cow, aux_c = self.cow_branch(x, cta_skips, perf_skips, aux4_3ch[:, 2:3])

        # 2. Nhánh Ischemic (Lesion + LVO) sử dụng đặc trưng Vascular để dẫn đường
        f_lesion, f_lvo, aux_i, g_maps = self.ischemic_branch(x, cta_skips, perf_skips, aux4_3ch[:, 0:2], vascular_feat=f_cow)

        # Gộp các Aux Masks
        aux_masks = [aux4_3ch]
        for i in range(3): # aux3, aux2, aux1
            # aux_i[i] là 2-channel (lesion, lvo), aux_c[i] là 1-channel (cow)
            mask = torch.cat([aux_i[i], aux_c[i]], dim=1)
            aux_masks.append(mask)

        features = {"lesion": f_lesion, "lvo": f_lvo, "cow": f_cow}
        return features, aux_masks, g_maps
