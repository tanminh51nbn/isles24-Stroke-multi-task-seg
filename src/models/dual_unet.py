"""
dual_unet.py — Module chính: Dual-Encoder Multi-Task UNet

Kiến trúc tổng thể:
    Input (18, 256, 256)
        ↓ Channel Split
    CTA (6ch) → ResNet-50 Encoder → [s1..s5]
    Perfusion (12ch) → DenseNet-121 Encoder → [d1..d5]
        ↓ Concatenate at each skip level
    UNet Triple Decoders → {f_lesion, f_lvo, f_cow}
        ↓ 3 Specialized Heads
    {lesion: (1,256,256), lvo: (1,256,256), cow: (1,256,256)}  — raw logits
"""

import torch
import torch.nn as nn
from typing import List

from models.encoder import ResNet50Encoder, DenseNet121Encoder
from models.decoder import MultiHeadDecoder
from models.heads import MultiTaskHeads


class DualEncoderUNet(nn.Module):
    """
    Dual-Encoder Multi-Task UNet cho bài toán Stroke Segmentation.

    Thiết kế:
        - 2 Encoder chuyên biệt (CTA + Perfusion) → tận dụng sở trường từng loại ảnh
        - 3 Decoder độc lập (Triple-Head Specialist) → tránh nhiễu đặc trưng giữa các task
        - 3 Heads độc lập → multi-task learning chuyên sâu
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

        from models.encoder import build_encoders
        self.cta_encoder, self.perf_encoder = build_encoders(config)

        # Decoder đa nhánh kết hợp skip features từ 2 encoder
        self.decoder = MultiHeadDecoder(config)

        # 3 heads độc lập
        # Lấy in_ch từ config của decoder (mặc định là 16)
        decoder_final_ch = config["decoder"].get("final_ch", 16)
        self.heads = MultiTaskHeads(
            in_ch=decoder_final_ch,
            dropout=config["heads"]["dropout"],
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, epoch: int = 0) -> dict:
        """
        Args:
            x: Tensor (B, 18, H, W)
            epoch: Epoch hiện tại (để điều khiển gating)
        Returns:
            dict: {
                'lesion': Tensor, 'lvo': Tensor, 'cow': Tensor,
                'aux_masks': [mask_32, mask_64, mask_128] (mỗi cái shape (B, 3, H_n, W_n))
            }
        """
        # Tách kênh
        cta_input  = x[:, self.cta_idx, :, :]
        perf_input = x[:, self.perf_idx, :, :]

        # Encode song song
        cta_skips  = self.cta_encoder(cta_input)
        perf_skips = self.perf_encoder(perf_input)

        # Decode với Iterative Feedback
        features, aux_masks, g_maps = self.decoder(cta_skips, perf_skips, epoch=epoch)

        # Multi-task output (Main heads)
        out = self.heads(features)
        
        # Đính kèm các masks phụ và guidance maps để phục vụ tính Loss/Debug
        out["aux_masks"] = aux_masks
        out["guidance_maps"] = g_maps
        
        return out

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
