"""
network.py — Multi-Task 2.5D Shared UNet for ISLES'24 Stroke Segmentation.

Architecture:
  - Shared Encoder: ResNet50 (loaded with RadImageNet, inflated to 18 channels via average-repeat)
  - Shared Decoder: UNet Decoder (smp.Unet)
  - 3 Independent Segmentation Heads: Lesion, LVO, CoW (raw logits)

Output:
  List of 3 tensors, each [B, 1, 544, 544] (raw logits)
  Order: [lesion, lvo, cow]
"""
import os
import logging
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead

logger = logging.getLogger(__name__)


class MultiTaskSharedUNet(nn.Module):
    """
    Multi-Task 2.5D Shared UNet for simultaneous Lesion/LVO/CoW segmentation.
    Shared encoder and decoder to learn task correlations, separate heads.
    """

    TASK_NAMES = ("lesion", "lvo", "cow")

    def __init__(self, cfg: dict):
        super().__init__()

        enc_cfg = cfg["encoder"]
        dec_cfg = cfg["decoder"]
        heads_cfg = cfg["heads"]

        # ════════════════════════════════════════════════════════
        #  Base UNet (Loaded with 3 channels to match RadImageNet)
        # ════════════════════════════════════════════════════════
        pretrained_weights = enc_cfg.get("weights", None)
        is_custom_weights = pretrained_weights and pretrained_weights.endswith(".pt")
        
        base_model = smp.Unet(
            encoder_name=enc_cfg["name"],
            encoder_weights=pretrained_weights if not is_custom_weights else None,
            in_channels=3,
            classes=1, # Dummy head, we will replace it
            decoder_channels=dec_cfg["channels"],
            decoder_attention_type=dec_cfg.get("attention_type", None),
        )

        self.encoder = base_model.encoder
        self.decoder = base_model.decoder

        # ════════════════════════════════════════════════════════
        #  Load RadImageNet Custom Weights
        # ════════════════════════════════════════════════════════
        if is_custom_weights:
            if os.path.exists(pretrained_weights):
                logger.info(f"Loading custom weights from {pretrained_weights}")
                # RadImageNet weights are standard torchvision ResNet50 keys.
                # SMP ResNet encoder uses the same keys.
                state_dict = torch.load(pretrained_weights, map_location="cpu")
                # Handle nested state_dicts (e.g., {'state_dict': ..., 'epoch': ...})
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]
                
                load_result = self.encoder.load_state_dict(state_dict, strict=False)
                
                # Safe unpacking for older PyTorch or modified modules
                if load_result is not None:
                    missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
                    logger.info(f"Custom weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                else:
                    logger.info("Custom weights loaded (load_state_dict returned None).")
            else:
                logger.warning(f"Custom weights file {pretrained_weights} NOT FOUND! Using random weights.")

        # ════════════════════════════════════════════════════════
        #  Inflate Conv1 (Average-Repeat)
        # ════════════════════════════════════════════════════════
        target_in_channels = enc_cfg["in_channels"]
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                old_weight = m.weight.data
                # w_new = w_old.mean(dim=1, keepdim=True).repeat(1, 18, 1, 1) / (18/3)
                new_weight = old_weight.mean(dim=1, keepdim=True).repeat(1, target_in_channels, 1, 1) / (target_in_channels / 3.0)
                m.in_channels = target_in_channels
                m.weight = nn.Parameter(new_weight)
                logger.info(f"Inflated first Conv2d from 3 to {target_in_channels} channels using Average-Repeat.")
                break

        # ════════════════════════════════════════════════════════
        #  3 Independent Segmentation Heads
        # ════════════════════════════════════════════════════════
        decoder_out_channels = dec_cfg["channels"][-1]
        self.heads = nn.ModuleDict()

        for task_name in self.TASK_NAMES:
            out_ch = heads_cfg[task_name]["out_channels"]
            head = SegmentationHead(
                in_channels=decoder_out_channels,
                out_channels=out_ch,
                activation=None,  # ⚠️ Raw logits!
                kernel_size=3,    # UNet default head
                upsampling=1,     # UNet decoder already upsamples to H,W
            )
            self.heads[task_name] = head

        self._init_heads()

    def _init_heads(self):
        """Kaiming initialization for task heads."""
        for m in self.heads.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass:
        Args: x: [B, 18, 544, 544]
        Returns: List of 3 raw logits [lesion, lvo, cow]
        """
        features = self.encoder(x)
        decoded = self.decoder(features)

        outputs = []
        for task_name in self.TASK_NAMES:
            outputs.append(self.heads[task_name](decoded))

        return outputs


def build_model(cfg: dict) -> MultiTaskSharedUNet:
    model = MultiTaskSharedUNet(cfg)

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    head_params = sum(p.numel() for p in model.heads.parameters())
    total_params = enc_params + dec_params + head_params

    logger.info(
        f"MultiTaskSharedUNet built: "
        f"encoder={enc_params / 1e6:.1f}M, "
        f"decoder={dec_params / 1e6:.1f}M, "
        f"heads={head_params / 1e6:.1f}M, "
        f"total={total_params / 1e6:.1f}M"
    )
    logger.info(f"  Encoder: {cfg['encoder']['name']} (depth={cfg['encoder']['depth']})")
    logger.info(f"  Tasks: {MultiTaskSharedUNet.TASK_NAMES}")
    return model
