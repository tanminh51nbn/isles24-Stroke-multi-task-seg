"""
dual_unet.py — Module chính: Dual-Encoder Multi-Task UNet

Kiến trúc tổng thể:
    Input (18, 256, 256)
        ↓ Channel Split
    CTA (6ch) → ResNet-50 Encoder → [s1..s5]
    Perfusion (12ch) → DenseNet-121 Encoder → [d1..d5]
        ↓ Concatenate at each skip level
    UNet Decoder → feature_map (16ch, 256, 256)
        ↓ 3 Heads
    {lesion: (1,256,256), lvo: (1,256,256), cow: (1,256,256)}  — raw logits
"""

import torch
import torch.nn as nn
from typing import List

from .encoder import ResNet50Encoder, DenseNet121Encoder
from .decoder import UNetDecoder
from .heads import MultiTaskHeads


class DualEncoderUNet(nn.Module):
    """
    Dual-Encoder Multi-Task UNet cho bài toán Stroke Segmentation.

    Thiết kế:
        - 2 Encoder chuyên biệt (CTA + Perfusion) → tận dụng sở trường từng loại ảnh
        - 1 Decoder chung → học cách kết hợp thông tin đa phương thức
        - 3 Heads độc lập → multi-task learning
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Dict đọc từ model.yaml
        """
        super().__init__()

        # Channel indices để tách input
        self.cta_idx  = config["channel_split"]["cta_indices"]    # 6 kênh
        self.perf_idx = config["channel_split"]["perfusion_indices"]  # 12 kênh

        # Encoder CTA: ResNet-50 + RadImageNet
        self.cta_encoder = ResNet50Encoder(
            in_channels=len(self.cta_idx),
            weights_path=config["cta_encoder"]["weights"],
        )

        # Encoder Perfusion: DenseNet-121 + RadImageNet
        self.perf_encoder = DenseNet121Encoder(
            in_channels=len(self.perf_idx),
            weights_path=config["perfusion_encoder"]["weights"],
        )

        # Decoder kết hợp skip features từ 2 encoder
        self.decoder = UNetDecoder()

        # 3 heads độc lập
        self.heads = MultiTaskHeads(
            in_ch=16,  # Output của decoder final_upsample
            dropout=config["heads"]["dropout"],
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Tensor (B, 18, H, W) — 18-channel 2.5D CT input

        Returns:
            dict {'lesion': ..., 'lvo': ..., 'cow': ...}
            Mỗi value là Tensor (B, 1, H, W) raw logits
        """
        # Tách kênh
        cta_input  = x[:, self.cta_idx, :, :]   # (B, 6,  H, W)
        perf_input = x[:, self.perf_idx, :, :]  # (B, 12, H, W)

        # Encode song song qua 2 backbone
        cta_skips  = self.cta_encoder(cta_input)   # [s1..s5]
        perf_skips = self.perf_encoder(perf_input)  # [d1..d5]

        # Decode
        features = self.decoder(cta_skips, perf_skips)  # (B, 16, H, W)

        # Multi-task output
        return self.heads(features)

    # ── Freeze / Unfreeze ────────────────────────────────────────────────────

    def freeze_encoders(self):
        """Đóng băng cả 2 encoder — dùng trong N epoch đầu để ổn định decoder."""
        for param in self.cta_encoder.parameters():
            param.requires_grad = False
        for param in self.perf_encoder.parameters():
            param.requires_grad = False
        print("[DualEncoderUNet] Encoders FROZEN")

    def unfreeze_encoders(self):
        """Mở băng encoder — bắt đầu fine-tune toàn bộ mạng."""
        for param in self.cta_encoder.parameters():
            param.requires_grad = True
        for param in self.perf_encoder.parameters():
            param.requires_grad = True
        print("[DualEncoderUNet] Encoders UNFROZEN — Full fine-tuning")

    # ── Param groups (Differential LR) ──────────────────────────────────────

    def get_param_groups(self, encoder_lr: float, decoder_lr: float) -> List[dict]:
        """
        Trả về param groups cho AdamW với Differential LR:
            - Encoder: encoder_lr (thấp hơn để bảo vệ RadImageNet weights)
            - Decoder + Heads: decoder_lr (cao hơn để học nhanh)
        """
        return [
            {
                "params": list(self.cta_encoder.parameters()) +
                          list(self.perf_encoder.parameters()),
                "lr": encoder_lr,
                "name": "encoders",
            },
            {
                "params": list(self.decoder.parameters()) +
                          list(self.heads.parameters()),
                "lr": decoder_lr,
                "name": "decoder_heads",
            },
        ]


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_model(config: dict) -> DualEncoderUNet:
    """
    Khởi tạo DualEncoderUNet từ config dict.

    Args:
        config: Dict đọc từ model.yaml

    Returns:
        model chưa được đưa lên GPU (gọi .to(device) hoặc DDP bên ngoài)
    """
    model = DualEncoderUNet(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DualEncoderUNet] Total params: {total_params:,} | Trainable: {trainable:,}")
    return model
