"""
single_unet.py — Module: Single-Encoder Multi-Task UNet (với Triple-Decoder Knowledge Cascade)

Kiến trúc tổng thể:
    Input (18, 256, 256) (Tất cả CTA + Perfusion)
        ↓
    DenseNet-121 Encoder (Early Fusion) → [s1..s5]
        ↓
    Triple-Decoder (Knowledge Cascade)
        ├── SharedPath (dec4, dec3)
        ├── TaskPath (CoW)
        ├── TaskPath (LVO, guidance=CoW)
        └── TaskPath (Lesion, guidance=CoW+Dropout(0.4))
        ↓
    MultiTaskHeads (Lesion, LVO, CoW)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from models.encoder import build_encoders
from models.decoder import ConvBnGelu1x1, ConvBnGelu, LightweightDualAttention, AttentionGate, AuxHead, TaskConditionedFiLM
from models.heads import MultiTaskHeads


# ─── Módulo Guidance Mềm (Spatial Attention) ──────────────────────────────────

class FusedSpatialAttention(nn.Module):
    """
    Kết hợp Đặc trưng của Task hiện tại và Đặc trưng Hướng dẫn (CoW) 
    để tự động sinh ra Bản đồ Không gian 1 kênh (Đèn pin) có khả năng Dập tắt (Masking)
    hoặc Tăng cường (Residual).
    """
    def __init__(self, task_ch: int, guidance_ch: int, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.attn_conv = nn.Sequential(
            nn.Conv2d(task_ch + guidance_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x_task, guidance_features):
        # --- Shape Assertion Mode (D2) ---
        from models.single_unet import SingleEncoderUNet
        if getattr(SingleEncoderUNet, 'DEBUG', False):
            assert x_task.shape[0] == guidance_features.shape[0], \
                f"[FusedSpatialAttention] Batch size mismatch: task={x_task.shape[0]}, guidance={guidance_features.shape[0]}."

        # 1. Resize guidance để khớp với x_task
        g_interp = F.interpolate(guidance_features, size=x_task.shape[2:], mode='bilinear', align_corners=False)
        # 2. Ghép nối để mạng nhìn thấy cả 2
        fused = torch.cat([x_task, g_interp], dim=1)
        # 3. Tạo bản đồ Đèn pin (0 đến 1)
        attn_map = self.attn_conv(fused)
        # 4. Áp dụng Attention (Residual giúp chống 'mù' nếu guidance thiếu sót)
        if self.residual:
            out = x_task + x_task * attn_map
        else:
            out = x_task * attn_map
        return out, attn_map


# ─── Khối Decoder cho 1 Encoder (Single Decoder Block) ─────────────────────────

class SingleDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention_type: Optional[str] = "dual", use_aux: bool = True, task_name: str = "shared", aux_ch: int = 1, dropout_p: float = 0.2):
        super().__init__()
        self.attention_type = attention_type
        self.use_aux = use_aux
        self.aux_ch = aux_ch
        self.task_name = task_name
        
        self.conv1 = ConvBnGelu1x1(in_ch + skip_ch + aux_ch, out_ch)
        self.conv2 = ConvBnGelu(out_ch, out_ch)
        self.dropout = nn.Dropout2d(p=dropout_p)
        
        if attention_type == "ag":
            self.ag = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=skip_ch // 2)
        elif attention_type == "dual":
            self.dual_attn = LightweightDualAttention(channels=skip_ch)
            
        if use_aux:
            self.aux_head = AuxHead(out_ch, task_name, out_ch=aux_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, prev_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        interp_mode = "nearest" if self.task_name == "lvo" else "bilinear"
        
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if prev_mask is None:
            prev_mask = torch.zeros((x_up.shape[0], self.aux_ch, x_up.shape[2], x_up.shape[3]), device=x.device)
        else:
            prev_mask = F.interpolate(prev_mask, size=(x_up.shape[2], x_up.shape[3]), mode=interp_mode, align_corners=(False if interp_mode == "bilinear" else None))
        
        # --- Shape Assertion Mode (D2) ---
        from models.single_unet import SingleEncoderUNet
        if getattr(SingleEncoderUNet, 'DEBUG', False):
            assert x_up.shape[2:] == skip.shape[2:], \
                f"[SingleDecoderBlock - {self.task_name}] Spatial shape mismatch between x_up {x_up.shape[2:]} and skip {skip.shape[2:]}."
            assert x_up.shape[2:] == prev_mask.shape[2:], \
                f"[SingleDecoderBlock - {self.task_name}] Spatial shape mismatch between x_up {x_up.shape[2:]} and prev_mask {prev_mask.shape[2:]}."
            assert x_up.shape[0] == skip.shape[0] == prev_mask.shape[0], \
                f"[SingleDecoderBlock - {self.task_name}] Batch size mismatch: x_up={x_up.shape[0]}, skip={skip.shape[0]}, prev_mask={prev_mask.shape[0]}."

        if self.attention_type == "ag":
            skip = self.ag(g=x_up, x=skip)
        elif self.attention_type == "dual":
            skip = self.dual_attn(skip)
            
        # --- Channel Dimension Assertion Mode (D2) ---
        if getattr(SingleEncoderUNet, 'DEBUG', False):
            expected_ch = x_up.shape[1] + skip.shape[1] + prev_mask.shape[1]
            conv1_in_ch = self.conv1.block[0].in_channels
            assert conv1_in_ch == expected_ch, \
                f"[SingleDecoderBlock - {self.task_name}] Conv1 expects {conv1_in_ch} input channels, but concat output has {expected_ch} channels."

        out = torch.cat([x_up, skip, prev_mask], dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.dropout(out)
        aux_out = self.aux_head(out) if self.use_aux else None
        return out, aux_out


# ─── Módulo SE Block & ASPP (Chống Bất đối xứng) ───────────────────────────────

class ModalitySEBlock(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DenseGlobalBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 7x7 Depthwise Conv để quét toàn cục (bao quát 8x8) mà không có lỗ hổng
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False)
        self.norm = nn.BatchNorm2d(in_channels)
        self.pwconv1 = nn.Conv2d(in_channels, 4 * in_channels, 1, bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * in_channels, out_channels, 1, bias=False)
        
    def forward(self, x):
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return res + x


class SafePerfusionBottleneck(nn.Module):
    """
    Bottleneck an toàn cho Perfusion:
    Chỉ áp dụng InstanceNorm cho các lát cắt (slices) thực sự có tín hiệu Perfusion.
    Ngăn chặn việc InstanceNorm khuếch đại nhiễu từ các lát cắt toàn số 0 (do zero variance).
    """
    def __init__(self, perf_ch: int, out_ch: int = 16):
        super().__init__()
        self.norm = nn.InstanceNorm2d(perf_ch, affine=True)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(perf_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        self.out_ch = out_ch

    def forward(self, perf_raw):
        B, C, H, W = perf_raw.shape
        eps = 1e-5
        
        # Detect lát cắt có tín hiệu perfusion (tổng trị tuyệt đối > eps)
        has_perf = (perf_raw.abs().sum(dim=[1, 2, 3]) > eps) # (B,)
        
        if not has_perf.any():
            # Nếu toàn batch trống, ta vẫn pass qua layer để Pytorch tự ép kiểu AMP (Half/Float)
            # sau đó nhân 0 để triệt tiêu hoàn toàn nhiễu (bias)
            dummy_out = self.bottleneck(self.norm(perf_raw))
            return dummy_out * 0.0
            
        valid_perf = perf_raw[has_perf]
        processed = self.bottleneck(self.norm(valid_perf))
        
        # Tạo out tensor KHỚP với dtype của processed (được AMP tự động cast)
        out = torch.zeros(B, self.out_ch, H, W, device=perf_raw.device, dtype=processed.dtype)
        out[has_perf] = processed
        
        return out


class HemisphericAsymmetryModule(nn.Module):
    """
    So sánh bất đối xứng 2 bán cầu não (chống nhiễu Head Tilt).
    Áp dụng ở độ phân giải 64x64 (tầng dec2) để khử nhiễu sai lệch vật lý.
    Sử dụng Depthwise Conv 21x21 (bán kính 10 pixels) để bắt các điểm bị lệch
    do bệnh nhân nghiêng đầu hoặc do Augmentation (Rotate/Translate).
    """
    def __init__(self, channels: int):
        super().__init__()
        # groups=channels với in_ch=2*channels giúp ghép cặp (feat_i, flipped_i)
        self.catch_conv = nn.Conv2d(
            in_channels=channels * 2, 
            out_channels=channels, 
            kernel_size=21, 
            padding=10, 
            groups=channels, 
            bias=False
        )
        self.norm = nn.BatchNorm2d(channels)
        self.gelu = nn.GELU()
        
        # Trộn thông tin bất đối xứng giữa các kênh với nhau
        self.mix_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # Cổng học được (Learnable Gate) để tự động điều chỉnh mức độ Asymmetry
        self.gate_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lật theo trục X (chiều ngang bán cầu)
        x_flipped = torch.flip(x, dims=[-1])
        
        # Nối x và x_flipped: (B, C*2, H, W)
        concat = torch.cat([x, x_flipped], dim=1)
        
        # Hứng lệch trục và tính Diff
        diff = self.catch_conv(concat)
        diff = self.norm(diff)
        diff = self.gelu(diff)
        
        # [FIX] High-Pass Spatial Filter: Chặn rò rỉ Lesion vào nhánh LVO
        # Các vùng Lesion tạo ra mảng bất đối xứng khổng lồ, trong khi LVO chỉ là đốm nhỏ.
        # Ta dùng AvgPool (Low-pass) để bắt mảng nền to, sau đó lấy tín hiệu gốc trừ đi mảng nền
        # -> Chỉ còn lại các đốm LVO sắc nét lọt qua.
        blur = F.avg_pool2d(diff, kernel_size=31, stride=1, padding=15)
        diff_sharp = diff - blur
        
        diff_out = self.mix_conv(diff_sharp)
        
        # Tính gating weight alpha (0 -> 1, khởi tạo ở 0.5)
        alpha = torch.sigmoid(self.gate_param)
        
        # Cộng feature bất đối xứng vào feature gốc với tỷ lệ alpha
        return x + alpha * diff_out


# ─── Specialized Decoder Paths (Single Encoder version) ────────────────────────

class SingleSharedPath(nn.Module):
    def __init__(self, config: dict, skip_channels: List[int]):
        super().__init__()
        dec_ch = config["decoder"]["out_channels"]
        attn_type = config["decoder"].get("attention_type", "dual")

        dropout_cfg = config["decoder"].get("dropout", {})
        dropout_p = dropout_cfg.get("shared", 0.2) if isinstance(dropout_cfg, dict) else (dropout_cfg if isinstance(dropout_cfg, float) else 0.2)

        # skip_channels = [s4, s3] (với s5 là bottleneck)
        self.dec4 = SingleDecoderBlock(1024, skip_channels[0], dec_ch[0], attn_type, use_aux=False, aux_ch=0, dropout_p=dropout_p)
        self.dec3 = SingleDecoderBlock(dec_ch[0], skip_channels[1], dec_ch[1], attn_type, use_aux=False, aux_ch=0, dropout_p=dropout_p)

    def forward(self, x_bottleneck, skips_shared):
        s4, s3 = skips_shared
        x, _ = self.dec4(x_bottleneck, s4, prev_mask=None)
        x, _ = self.dec3(x, s3, prev_mask=None)
        return x


class SingleTaskPath(nn.Module):
    def __init__(self, in_ch: int, config: dict, task_name: str, skip_channels: List[int],
                 aux_ch: int = 1, active_aux_levels: List[bool] = [True, True],
                 guidance_ch: int = 0, guidance_dec2_ch: int = 0, guidance_dec1_ch: int = 0):
        super().__init__()
        self.task_name = task_name
        dec_ch = config["decoder"]["out_channels"]
        final_ch = config["decoder"].get("final_ch", 16)
        attn_type = config["decoder"].get("attention_type", "dual")

        dropout_cfg = config["decoder"].get("dropout", {})
        dropout_p = dropout_cfg.get(task_name, 0.2) if isinstance(dropout_cfg, dict) else (dropout_cfg if isinstance(dropout_cfg, float) else 0.2)

        self.dec2 = SingleDecoderBlock(in_ch, skip_channels[0], dec_ch[2], attn_type, use_aux=active_aux_levels[0], task_name=task_name, aux_ch=aux_ch, dropout_p=dropout_p)
        self.dec1 = SingleDecoderBlock(dec_ch[2], skip_channels[1], dec_ch[3], attn_type, use_aux=active_aux_levels[1], task_name=task_name, aux_ch=aux_ch, dropout_p=dropout_p)

        # ─── Task-Conditioned FiLM ──────────────────────────────────────────
        self.task_embedding = nn.Parameter(torch.randn(1, 64))
        self.film_shared = TaskConditionedFiLM(in_channels=in_ch, embedding_dim=64)
        self.film_s2 = TaskConditionedFiLM(in_channels=skip_channels[0], embedding_dim=64)
        self.film_s1 = TaskConditionedFiLM(in_channels=skip_channels[1], embedding_dim=64)

        # ─── Hemispheric Asymmetry Module ──────────────────────────────────
        # CHỈ áp dụng cho LVO. Không dùng cho Lesion (nhồi máu mờ, vắt ngang, đồi thị)
        # Không dùng cho CoW (đối xứng tự nhiên).
        if task_name == "lvo":
            self.asymmetry_module = HemisphericAsymmetryModule(dec_ch[2])
        else:
            self.asymmetry_module = None

        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = ConvBnGelu(dec_ch[3], final_ch)
        
        if guidance_ch > 0:
            self.guidance_attn = FusedSpatialAttention(task_ch=final_ch, guidance_ch=guidance_ch)
        else:
            self.guidance_attn = None

        if guidance_dec2_ch > 0:
            self.attn_dec2 = FusedSpatialAttention(task_ch=dec_ch[2], guidance_ch=guidance_dec2_ch)
        else:
            self.attn_dec2 = None

        if guidance_dec1_ch > 0:
            self.attn_dec1 = FusedSpatialAttention(task_ch=dec_ch[3], guidance_ch=guidance_dec1_ch)
        else:
            self.attn_dec1 = None

    def forward(self, x_shared, skips_task, guidance: Optional[torch.Tensor] = None,
                guidance_dec2: Optional[torch.Tensor] = None, guidance_dec1: Optional[torch.Tensor] = None):
        s2, s1 = skips_task
        
        # ─── Áp dụng FiLM ──────────────────────────────────────────────────
        x_shared_film = self.film_shared(x_shared, self.task_embedding)
        s2_film = self.film_s2(s2, self.task_embedding)
        s1_film = self.film_s1(s1, self.task_embedding)

        x_dec2, aux2 = self.dec2(x_shared_film, s2_film, prev_mask=None)
        if guidance_dec2 is not None and self.attn_dec2 is not None:
            x_dec2, _ = self.attn_dec2(x_dec2, guidance_dec2)

        # Áp dụng Asymmetry Module tại tầng dec2 (64x64)
        if self.asymmetry_module is not None:
            x_dec2 = self.asymmetry_module(x_dec2)

        x_dec1, aux1 = self.dec1(x_dec2, s1_film, prev_mask=aux2)
        if guidance_dec1 is not None and self.attn_dec1 is not None:
            x_dec1, _ = self.attn_dec1(x_dec1, guidance_dec1)

        x = self.up_final(x_dec1)
        x = self.final_conv(x)
        
        if guidance is not None and self.guidance_attn is not None:
            x, attn_map = self.guidance_attn(x, guidance)

        return x, [None, None, aux2, aux1], [x_dec2, x_dec1]


class LesionTaskPath(nn.Module):
    def __init__(self, in_ch: int, config: dict, skip_channels: List[int], perf_ch: int = 6,
                 guidance_dec2_ch: int = 0, guidance_dec1_ch: int = 0):
        super().__init__()
        dec_ch = config["decoder"]["out_channels"]
        final_ch = config["decoder"].get("final_ch", 16)
        attn_type = config["decoder"].get("attention_type", "dual")
        
        dropout_cfg = config["decoder"].get("dropout", {})
        dropout_p = dropout_cfg.get("lesion", 0.2) if isinstance(dropout_cfg, dict) else 0.2
        
        # ─── Safe Perfusion Bottleneck (Xử lý an toàn lát cắt trống) ──────────
        self.perf_bottleneck = SafePerfusionBottleneck(perf_ch=perf_ch, out_ch=16)
        
        # Bơm Perfusion Bottleneck (16 kênh) vào skip channels
        self.dec2 = SingleDecoderBlock(in_ch, skip_channels[0] + 16, dec_ch[2], attn_type, use_aux=True, task_name="lesion", aux_ch=1, dropout_p=dropout_p)
        self.dec1 = SingleDecoderBlock(dec_ch[2], skip_channels[1] + 16, dec_ch[3], attn_type, use_aux=True, task_name="lesion", aux_ch=1, dropout_p=dropout_p)

        # ─── Task-Conditioned FiLM ──────────────────────────────────────────
        self.task_embedding = nn.Parameter(torch.randn(1, 64))
        self.film_shared = TaskConditionedFiLM(in_channels=in_ch, embedding_dim=64)
        self.film_s2 = TaskConditionedFiLM(in_channels=skip_channels[0], embedding_dim=64)
        self.film_s1 = TaskConditionedFiLM(in_channels=skip_channels[1], embedding_dim=64)
        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        # dec_ch[3] + 1 vì ta sẽ nối thêm perf_mask (1 kênh)
        self.final_conv = ConvBnGelu(dec_ch[3] + 1, final_ch)
        self.guidance_attn = FusedSpatialAttention(task_ch=final_ch, guidance_ch=16)

        if guidance_dec2_ch > 0:
            self.attn_dec2 = FusedSpatialAttention(task_ch=dec_ch[2], guidance_ch=guidance_dec2_ch)
        else:
            self.attn_dec2 = None

        if guidance_dec1_ch > 0:
            self.attn_dec1 = FusedSpatialAttention(task_ch=dec_ch[3], guidance_ch=guidance_dec1_ch)
        else:
            self.attn_dec1 = None

    def forward(self, x_shared, skips_task, guidance, perf_raw,
                guidance_dec2: Optional[torch.Tensor] = None, guidance_dec1: Optional[torch.Tensor] = None):
        s2, s1 = skips_task
        
        # ─── Perfusion Bottleneck ─────────────────────────────────────────
        # perf_raw is (B, 6, 256, 256)
        perf_norm = self.perf_bottleneck(perf_raw) # (B, 16, 256, 256)
        
        # s2 is 64x64, s1 is 128x128
        perf_s2 = F.avg_pool2d(perf_norm, 4)
        perf_s1 = F.avg_pool2d(perf_norm, 2)
        
        # ─── Áp dụng FiLM ──────────────────────────────────────────────────
        x_shared_film = self.film_shared(x_shared, self.task_embedding)
        s2_film = self.film_s2(s2, self.task_embedding)
        s1_film = self.film_s1(s1, self.task_embedding)
        
        s2_fused = torch.cat([s2_film, perf_s2], dim=1)
        s1_fused = torch.cat([s1_film, perf_s1], dim=1)
        
        x_dec2, aux2 = self.dec2(x_shared_film, s2_fused, prev_mask=None)
        if guidance_dec2 is not None and self.attn_dec2 is not None:
            x_dec2, _ = self.attn_dec2(x_dec2, guidance_dec2)

        x_dec1, aux1 = self.dec1(x_dec2, s1_fused, prev_mask=aux2)
        if guidance_dec1 is not None and self.attn_dec1 is not None:
            x_dec1, _ = self.attn_dec1(x_dec1, guidance_dec1)

        x = self.up_final(x_dec1)
        
        # ─── Bơm tường minh (Explicit Mask) thông tin Perfusion vào Lesion Head ───
        # Để mô hình tự động fallback sang CTA nếu không có Perfusion
        has_perf = (perf_raw.abs().sum(dim=[1,2,3]) > 1e-5).float() # (B,)
        perf_mask = has_perf.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3]) # (B, 1, H, W)
        
        x = torch.cat([x, perf_mask], dim=1) # (B, dec_ch[3] + 1, H, W)
        x = self.final_conv(x)
        
        if guidance is not None and self.guidance_attn is not None:
            x, _ = self.guidance_attn(x, guidance)
            
        return x, [None, None, aux2, aux1], [x_dec2, x_dec1]


class SingleEncoderTripleDecoder(nn.Module):
    def __init__(self, config: dict, skip_channels: List[int]):
        super().__init__()
        bottleneck_ch = 1024
        
        # skip_channels = [ch_s1, ch_s2, ch_s3, ch_s4, ch_s5]
        
        self.shared_bottleneck = nn.Sequential(
            ConvBnGelu1x1(skip_channels[4], bottleneck_ch),
            DenseGlobalBottleneck(bottleneck_ch, bottleneck_ch),
        )

        skips_shared = [skip_channels[3], skip_channels[2]]  # s4, s3
        skips_task   = [skip_channels[1], skip_channels[0]]  # s2, s1
        
        self.shared_path = SingleSharedPath(config, skips_shared)
        
        dec_ch = config["decoder"]["out_channels"]
        in_ch_task = dec_ch[1] # output of dec3
        
        self.cow_path    = SingleTaskPath(in_ch_task, config, "cow", skips_task, aux_ch=1, active_aux_levels=[True, True], guidance_ch=0)
        
        self.lvo_path = SingleTaskPath(in_ch_task, config, "lvo", skips_task, aux_ch=1, active_aux_levels=[False, False], guidance_ch=16,
                                       guidance_dec2_ch=64, guidance_dec1_ch=32)
        
        self.lesion_path = LesionTaskPath(in_ch_task, config, skips_task, perf_ch=6,
                                          guidance_dec2_ch=64, guidance_dec1_ch=32)
        
        # Dropout 2D để "cai nghiện" sự phụ thuộc của Lesion vào LVO/CoW (Giảm 0.4 → 0.15 tránh phá hủy cascade)
        self.guidance_dropout = nn.Dropout2d(p=0.2)

    def forward(self, skips: List[torch.Tensor], epoch: int = 0, x_raw: Optional[torch.Tensor] = None, decoupled: bool = False):
        # --- Shape Assertion Mode (D2) ---
        from models.single_unet import SingleEncoderUNet
        if getattr(SingleEncoderUNet, 'DEBUG', False):
            assert len(skips) == 5, f"[SingleEncoderTripleDecoder] Expected 5 skip tensors from encoder, got {len(skips)}."
            # s1 to s5 channels: s1 should be 64, s2=256, s3=512, s4=1024, s5=1024 (DenseNet-121 skips)
            expected_channels = [64, 256, 512, 1024, 1024]
            for idx, s in enumerate(skips):
                assert s.shape[1] == expected_channels[idx], \
                    f"[SingleEncoderTripleDecoder] Encoder skip s{idx+1} expects {expected_channels[idx]} channels, got {s.shape[1]}."
                expected_size = skips[0].shape[2] // (2 ** idx)
                assert s.shape[2] == expected_size and s.shape[3] == expected_size, \
                    f"[SingleEncoderTripleDecoder] Encoder skip s{idx+1} expects spatial size {(expected_size, expected_size)}, got {s.shape[2:]}."
        
        s1, s2, s3, s4, s5 = skips

        if decoupled and self.training:
            # 1. Bottleneck with detached leaves for shared path inputs
            s5_dec = s5.detach().requires_grad_(True)
            s4_dec = s4.detach().requires_grad_(True)
            s3_dec = s3.detach().requires_grad_(True)
            
            x_bottleneck = self.shared_bottleneck(s5_dec)
            x_shared = self.shared_path(x_bottleneck, [s4_dec, s3_dec])
            
            # Save detached leaves for backward access
            self.s5_dec = s5_dec
            self.s4_dec = s4_dec
            self.s3_dec = s3_dec
            self.x_shared = x_shared
            
            # 2. Detach x_shared and skips for each task path to isolate their graphs
            x_shared_cow = x_shared.detach().requires_grad_(True)
            s2_cow = s2.detach().requires_grad_(True)
            s1_cow = s1.detach().requires_grad_(True)
            
            x_shared_lvo = x_shared.detach().requires_grad_(True)
            s2_lvo = s2.detach().requires_grad_(True)
            s1_lvo = s1.detach().requires_grad_(True)
            
            x_shared_les = x_shared.detach().requires_grad_(True)
            s2_les = s2.detach().requires_grad_(True)
            s1_les = s1.detach().requires_grad_(True)
            
            self.task_leaves = {
                "cow": (x_shared_cow, s2_cow, s1_cow),
                "lvo": (x_shared_lvo, s2_lvo, s1_lvo),
                "lesion": (x_shared_les, s2_les, s1_les)
            }
            
            f_cow, cow_auxs, cow_feats = self.cow_path(x_shared_cow, [s2_cow, s1_cow])
            cow_dec2, cow_dec1 = cow_feats
            
            # --- LVO nhận Guidance từ CoW ---
            guidance_for_lvo = f_cow.detach()
            f_lvo, lvo_auxs, _ = self.lvo_path(
                x_shared_lvo, [s2_lvo, s1_lvo],
                guidance=guidance_for_lvo,
                guidance_dec2=cow_dec2.detach(),
                guidance_dec1=cow_dec1.detach()
            )
            
            # --- Lesion chỉ nhận Guidance từ CoW (Mạch máu sạch) ---
            guidance_for_lesion = f_cow.detach()
            guidance_for_lesion = self.guidance_dropout(guidance_for_lesion)
            
            # [FIX #3] Không dropout deep guidance levels — cần tín hiệu sạch cho deep supervision
            cow_dec2_for_les = cow_dec2.detach()
            cow_dec1_for_les = cow_dec1.detach()
            
            perf_raw = x_raw[:, 6:12, :, :] if x_raw is not None else torch.zeros((s1.shape[0], 6, s1.shape[2]*2, s1.shape[3]*2), device=s1.device)
            f_lesion, lesion_auxs, _ = self.lesion_path(
                x_shared_les, [s2_les, s1_les],
                guidance=guidance_for_lesion,
                perf_raw=perf_raw,
                guidance_dec2=cow_dec2_for_les,
                guidance_dec1=cow_dec1_for_les
            )
        else:
            # 1. Bottleneck
            x_bottleneck = self.shared_bottleneck(s5)
            
            # 2. Shared Path (dec4, dec3)
            x_shared = self.shared_path(x_bottleneck, [s4, s3])
            
            f_cow, cow_auxs, cow_feats = self.cow_path(x_shared, [s2, s1])
            cow_dec2, cow_dec1 = cow_feats
            
            # --- LVO nhận Guidance từ CoW ---
            guidance_for_lvo = f_cow.detach()
            if self.training:
                guidance_for_lvo.requires_grad_(True)
                def lvo_guidance_hook(grad):
                    self._lvo_guidance_grad_norm = grad.norm(2).item()
                guidance_for_lvo.register_hook(lvo_guidance_hook)
                
            f_lvo, lvo_auxs, _ = self.lvo_path(
                x_shared, [s2, s1],
                guidance=guidance_for_lvo,
                guidance_dec2=cow_dec2.detach(),
                guidance_dec1=cow_dec1.detach()
            )
            
            # --- Lesion chỉ nhận Guidance từ CoW (Mạch máu sạch) ---
            guidance_for_lesion = f_cow.detach()
            guidance_for_lesion = self.guidance_dropout(guidance_for_lesion)
            
            if self.training:
                guidance_for_lesion.requires_grad_(True)
                def lesion_guidance_hook(grad):
                    self._lesion_guidance_grad_norm = grad.norm(2).item()
                guidance_for_lesion.register_hook(lesion_guidance_hook)
                
            # [FIX #3] Không dropout deep guidance levels — cần tín hiệu sạch cho deep supervision
            cow_dec2_for_les = cow_dec2.detach()
            cow_dec1_for_les = cow_dec1.detach()
            
            # Truyền raw perfusion vào Lesion Path
            perf_raw = x_raw[:, 6:12, :, :] if x_raw is not None else torch.zeros((s1.shape[0], 6, s1.shape[2]*2, s1.shape[3]*2), device=s1.device)
            f_lesion, lesion_auxs, _ = self.lesion_path(
                x_shared, [s2, s1],
                guidance=guidance_for_lesion,
                perf_raw=perf_raw,
                guidance_dec2=cow_dec2_for_les,
                guidance_dec1=cow_dec1_for_les
            )

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


# ─── Mô hình chính ─────────────────────────────────────────────────────────────

class SingleEncoderUNet(nn.Module):
    """
    Single-Encoder Multi-Task UNet với cơ chế Triple Decoder Knowledge Cascade
    """
    DEBUG = False

    def __init__(self, config: dict):
        super().__init__()
        
        if "encoder" in config:
            enc_cfg = config["encoder"]
        else:
            enc_cfg = config.get("cta_encoder")
            
        name = enc_cfg["name"]
        in_ch = enc_cfg["in_channels"]
        weights = enc_cfg.get("weights", None)
        drop = enc_cfg.get("enc_dropout", None)

        if name == "resnet50":
            from models.encoder import ResNet50Encoder
            self.encoder = ResNet50Encoder(in_channels=in_ch, weights_path=weights, enc_dropout=drop)
        elif name == "densenet121":
            from models.encoder import DenseNet121Encoder
            self.encoder = DenseNet121Encoder(in_channels=in_ch, weights_path=weights, enc_dropout=drop)
        else:
            raise ValueError(f"SingleEncoderUNet: Unknown encoder {name}")

        self.decoder = SingleEncoderTripleDecoder(
            config,
            skip_channels=self.encoder.skip_channels
        )

        decoder_final_ch = config["decoder"].get("final_ch", 16)
        self.heads = MultiTaskHeads(
            in_ch=decoder_final_ch,
            heads_config=config["heads"],
        )

    def forward(self, x: torch.Tensor, epoch: int = 0, decoupled: bool = False) -> dict:
        # --- Shape Assertion Mode (D2) ---
        if getattr(self, 'DEBUG', False):
            assert x.ndim == 4, f"[SingleEncoderUNet] Input must be a 4D tensor (B, C, H, W), got {x.shape}."
            assert x.shape[1] == 18, f"[SingleEncoderUNet] Input must have 18 channels (6 CTA + 12 CTP), got {x.shape[1]}."
            assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0, \
                f"[SingleEncoderUNet] Input dimensions must be multiple of 32 (for UNet downsampling), got {x.shape[2:]}."

        skips = self.encoder(x)
        if self.training:
            self.encoder.saved_skips = skips

        features_dict, aux_masks, g_maps = self.decoder(skips, epoch=epoch, x_raw=x, decoupled=decoupled)

        out = self.heads(features_dict)
        
        out["aux_masks"] = aux_masks
        out["guidance_maps"] = g_maps
        
        return out

    def freeze_encoders(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[SingleEncoderUNet] Encoder FROZEN")

    def unfreeze_encoders(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("[SingleEncoderUNet] Encoder UNFROZEN")

    def get_param_groups(self, encoder_lr: float, decoder_lr: float) -> List[dict]:
        film_params = []
        gate_params = []
        decoder_heads_params = []
        
        for name, p in list(self.decoder.named_parameters()) + list(self.heads.named_parameters()):
            if "film" in name.lower() or "embedding" in name.lower():
                film_params.append(p)
            elif "gate_param" in name.lower():
                gate_params.append(p)
            else:
                decoder_heads_params.append(p)
                
        return [
            {
                "params": list(self.encoder.parameters()),
                "lr": encoder_lr,
                "name": "encoders",
            },
            {
                "params": film_params,
                "lr": decoder_lr * 3.0,
                "name": "film",
            },
            {
                "params": gate_params,
                "lr": decoder_lr * 3.0,
                "name": "asymmetry_gate",
            },
            {
                "params": decoder_heads_params,
                "lr": decoder_lr,
                "name": "decoder_heads",
            },
        ]

def build_model(config: dict) -> nn.Module:
    """Tự động trả về mô hình tương ứng"""
    if "perfusion_encoder" in config:
        from models.dual_unet import DualEncoderUNet
        model = DualEncoderUNet(config)
        model_name = "DualEncoderUNet"
    else:
        model = SingleEncoderUNet(config)
        model_name = "SingleEncoderUNet"
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Total params: {total_params:,} | Trainable: {trainable:,}")
    return model
