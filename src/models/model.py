import logging

import segmentation_models_pytorch as smp
import torch.nn as nn

from .backbone import inflate_conv1, load_rad_weights
from .heads import MultiTaskHeads

logger = logging.getLogger(__name__)


class MultiTaskSharedUNet(nn.Module):
    """
    Multi-Task 2.5D Shared UNet.

    Pipeline:
        Input [B, 18, H, W]
            → ResNet50 Encoder (RadImageNet pretrained, conv1 inflated 3→18ch)
            → Shared UNet Decoder (SMP, skip connections từ encoder)
            → 3 Task Heads (Conv1×1): Lesion, LVO, CoW
            → 3× [B, 1, H, W] raw logits

    Lý do tạo 1 smp.Unet() duy nhất:
        SMP encoder và decoder được thiết kế để hoạt động cùng nhau.
        Tạo riêng lẻ có thể gây mismatch ở skip connection channels.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        enc_cfg  = cfg.get("encoder", {})
        dec_cfg  = cfg.get("decoder", {})
        head_cfg = cfg.get("heads", {})

        encoder_name     = enc_cfg.get("name", "resnet50")
        in_channels      = enc_cfg.get("in_channels", 18)
        rad_weight_path  = enc_cfg.get("weights", None)
        decoder_channels = dec_cfg.get("channels", [256, 128, 64, 32, 16])
        attention_type   = dec_cfg.get("attention_type", None)

        # ── Bước 1: Tạo 1 UNet duy nhất với in_channels=3 (SMP default) ──
        # Encoder và decoder được tạo cùng nhau → skip connections guaranteed match
        base = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,        # Không dùng ImageNet, sẽ load RadImageNet thủ công
            in_channels=3,               # Tạm thời 3ch, sẽ inflate sau
            classes=1,                   # Placeholder, không dùng segmentation_head
            decoder_channels=decoder_channels,
            decoder_attention_type=attention_type,
            decoder_use_batchnorm=True,
        )

        self.encoder = base.encoder
        self.decoder = base.decoder
        # Bỏ segmentation_head của SMP — thay bằng MultiTaskHeads
        # base.segmentation_head không được assign vào self

        # ── Bước 2: Load RadImageNet weights (trước khi inflate) ──
        load_rad_weights(self.encoder, rad_weight_path)

        # ── Bước 3: Inflate conv1 từ 3 → in_channels ──
        inflate_conv1(self.encoder, in_channels)

        # ── Bước 4: Build 3 task heads ──
        decoder_out_channels = decoder_channels[-1]  # = 16 (last element)
        self.heads = MultiTaskHeads(
            in_channels=decoder_out_channels,
            tasks_cfg=head_cfg,
        )

        logger.info(
            f"MultiTaskSharedUNet initialized: "
            f"encoder={encoder_name}, in_channels={in_channels}, "
            f"decoder_channels={decoder_channels}"
        )

    def forward(self, x):
        # features = [stem_out, layer1, layer2, layer3, layer4]
        # SMP decoder tự handle skip connections từ features list
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        out_lesion, out_lvo, out_cow = self.heads(decoder_output)
        return out_lesion, out_lvo, out_cow


def build_model(cfg: dict) -> nn.Module:
    """Factory function for MultiTaskSharedUNet."""
    return MultiTaskSharedUNet(cfg)
