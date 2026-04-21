"""
network.py — Multi-Task 2.5D U-Net for ISLES'24 Stroke Segmentation.

Architecture:
  - Shared Encoder: EfficientNet-B2 (18 input channels)
  - 3 Independent UnetDecoders: Lesion, LVO, CoW
  - 3 SegmentationHeads: raw logits output (no sigmoid)

Output:
  List of 3 tensors, each [B, 1, 544, 544] (raw logits)
  Order: [lesion, lvo, cow]

Medical design rationale:
  - 3 independent decoders allow each task to learn task-specific
    skip connection usage (Lesion=large regions, LVO=tiny points, CoW=tubular)
  - Raw logits output → BCEWithLogitsLoss handles sigmoid internally
    for better gradient flow and numerical stability
"""
import logging

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead

logger = logging.getLogger(__name__)


class MultiTaskUNet(nn.Module):
    """
    Multi-Task 2.5D U-Net for simultaneous Lesion/LVO/CoW segmentation.

    Hard Parameter Sharing: one encoder → three independent decoders.
    Each decoder has its own skip connections and segmentation head.
    """

    TASK_NAMES = ("lesion", "lvo", "cow")

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: model config dict (from configs/model.yaml)
        """
        super().__init__()

        enc_cfg = cfg["encoder"]
        dec_cfg = cfg["decoder"]
        heads_cfg = cfg["heads"]

        # ════════════════════════════════════════════════════════
        #  Shared Encoder
        # ════════════════════════════════════════════════════════
        self.encoder = smp.encoders.get_encoder(
            name=enc_cfg["name"],
            in_channels=enc_cfg["in_channels"],     # 18
            depth=enc_cfg["depth"],                  # 5
            weights=enc_cfg.get("weights"),          # "imagenet" or None
        )

        # Encoder output channels at each stage
        # e.g. EfficientNet-B2: (3, 32, 16, 24, 48, 120, 352) for depth=5
        encoder_channels = self.encoder.out_channels

        decoder_channels = tuple(dec_cfg["channels"])   # (256, 128, 64, 32, 16)
        n_blocks = len(decoder_channels)
        attention_type = dec_cfg.get("attention_type")   # None or "scse"
        center_block = dec_cfg.get("center_block", False)

        # ════════════════════════════════════════════════════════
        #  3 Independent Decoders
        # ════════════════════════════════════════════════════════
        # Each decoder learns its own skip connection weights,
        # allowing task-specific feature selection at each scale.
        for task_name in self.TASK_NAMES:
            decoder = UnetDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                n_blocks=n_blocks,
                attention_type=attention_type,
                add_center_block=center_block,
            )
            setattr(self, f"decoder_{task_name}", decoder)

        # ════════════════════════════════════════════════════════
        #  3 Segmentation Heads (raw logits, NO activation)
        # ════════════════════════════════════════════════════════
        for task_name in self.TASK_NAMES:
            out_ch = heads_cfg[task_name]["out_channels"]   # 1
            head = SegmentationHead(
                in_channels=decoder_channels[-1],   # 16
                out_channels=out_ch,                 # 1
                activation=None,                     # ⚠️ Raw logits!
                kernel_size=3,                       # 3×3 conv for smoothing
                upsampling=1,                        # No additional upsampling
            )
            setattr(self, f"head_{task_name}", head)

        # Initialize decoder/head weights
        self._init_decoder_heads()

    def _init_decoder_heads(self):
        """Kaiming initialization for decoder and head weights."""
        for task_name in self.TASK_NAMES:
            decoder = getattr(self, f"decoder_{task_name}")
            head = getattr(self, f"head_{task_name}")

            for module in [decoder, head]:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu"
                        )
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass: encode once, decode three times.

        Args:
            x: Input tensor [B, 18, 544, 544] float32

        Returns:
            List of 3 tensors (raw logits), each [B, 1, 544, 544]
            Order: [lesion_logits, lvo_logits, cow_logits]
        """
        # ── Shared encoding (computed ONCE) ──
        features = self.encoder(x)

        # ── Independent decoding per task ──
        outputs = []
        for task_name in self.TASK_NAMES:
            decoder = getattr(self, f"decoder_{task_name}")
            head = getattr(self, f"head_{task_name}")

            decoded = decoder(*features)
            logits = head(decoded)
            outputs.append(logits)

        return outputs   # [lesion, lvo, cow]


# ═══════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════

def build_model(cfg: dict) -> MultiTaskUNet:
    """
    Build MultiTaskUNet from config and log architecture details.

    Args:
        cfg: model config dict (from configs/model.yaml)

    Returns:
        Initialized MultiTaskUNet ready for training
    """
    model = MultiTaskUNet(cfg)

    # ── Log parameter counts ──
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    dec_head_params = total_params - enc_params

    logger.info(
        f"MultiTaskUNet built: "
        f"encoder={enc_params / 1e6:.1f}M, "
        f"decoders+heads={dec_head_params / 1e6:.1f}M, "
        f"total={total_params / 1e6:.1f}M"
    )
    logger.info(f"  Encoder: {cfg['encoder']['name']} (depth={cfg['encoder']['depth']})")
    logger.info(f"  Encoder out_channels: {model.encoder.out_channels}")
    logger.info(f"  Decoder channels: {cfg['decoder']['channels']}")
    logger.info(f"  Tasks: {MultiTaskUNet.TASK_NAMES}")
    logger.info(f"  Output: 3 × [B, 1, H, W] raw logits (no activation)")

    return model
