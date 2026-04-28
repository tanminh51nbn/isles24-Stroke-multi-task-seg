"""
encoder.py — Dual-Encoder: ResNet-50 (CTA) + DenseNet-121 (Perfusion)

Nhiệm vụ:
    - Load RadImageNet pre-trained weights cho từng backbone
    - Inflate lớp conv đầu tiên từ 3 kênh lên N kênh (Conv1 Inflation)
    - Trả về feature extractor với 5 cấp skip connections

Kỹ thuật Conv1 Inflation:
    W_new = repeat(mean(W_old, dim=1, keepdim=True), N, dim=1) × (3/N)
    Bảo toàn phương sai kích hoạt ban đầu (variance preservation).
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List


# ─── Utility: Conv1 Inflation ────────────────────────────────────────────────

def inflate_weights(weight: torch.Tensor, target_channels: int) -> torch.Tensor:
    """
    Thổi phồng weight conv đầu từ 3 kênh sang target_channels kênh.
    Công thức: mean → repeat → scale để bảo toàn variance.

    Args:
        weight: Tensor shape (out_ch, 3, kH, kW) — weight gốc 3 kênh
        target_channels: Số kênh đầu vào mới

    Returns:
        Tensor shape (out_ch, target_channels, kH, kW)
    """
    # Mean qua trục kênh → (out_ch, 1, kH, kW)
    mean_weight = weight.mean(dim=1, keepdim=True)
    # Repeat → (out_ch, target_channels, kH, kW)
    new_weight = mean_weight.repeat(1, target_channels, 1, 1)
    # Scale để bảo toàn phương sai
    new_weight = new_weight * (3.0 / target_channels)
    return new_weight


# ─── ResNet-50 Encoder (Nhánh CTA — 6 kênh) ─────────────────────────────────

class ResNet50Encoder(nn.Module):
    """
    ResNet-50 encoder trích xuất 5 cấp skip features cho UNet decoder.

    Skip output channels: [64, 256, 512, 1024, 2048]
    Input resolution 256×256 → feature maps: [128, 64, 32, 16, 8]
    """

    def __init__(self, in_channels: int = 6, weights_path: str = None):
        super().__init__()
        # Load backbone chuẩn (ImageNet) để lấy cấu trúc
        backbone = models.resnet50(weights=None)

        # Inflate conv1: (64, 3, 7, 7) → (64, in_channels, 7, 7)
        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels,
            old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )

        # Khởi tạo weight bằng inflation trước khi load RadImageNet
        with torch.no_grad():
            new_conv1.weight.copy_(
                inflate_weights(old_conv1.weight, in_channels)
            )
        backbone.conv1 = new_conv1

        # Load RadImageNet weights (nếu có)
        if weights_path is not None:
            self._load_radimagenet(backbone, weights_path, in_channels)

        # Tách thành 5 stage để lấy skip connections
        self.stage0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64ch, /2
        self.pool   = backbone.maxpool                                              # /4 tổng
        self.stage1 = backbone.layer1   # 256ch,  /4
        self.stage2 = backbone.layer2   # 512ch,  /8
        self.stage3 = backbone.layer3   # 1024ch, /16
        self.stage4 = backbone.layer4   # 2048ch, /32

    def _load_radimagenet(self, backbone: nn.Module, path: str, in_channels: int):
        """Load RadImageNet checkpoint, skip conv1 (đã inflate riêng)."""
        state_dict = torch.load(path, map_location="cpu")
        # Một số checkpoint có wrapper key
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Loại bỏ key của conv1 để giữ weight inflation
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("conv1")}
        backbone.load_state_dict(state_dict, strict=False)
        print(f"[ResNet50Encoder] Loaded RadImageNet weights from: {path}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            List 5 feature maps từ nông → sâu:
            [s1(64ch), s2(256ch), s3(512ch), s4(1024ch), s5(2048ch)]
        """
        s1 = self.stage0(x)        # (B, 64,  H/2,  W/2)
        x  = self.pool(s1)         # (B, 64,  H/4,  W/4)
        s2 = self.stage1(x)        # (B, 256, H/4,  W/4)
        s3 = self.stage2(s2)       # (B, 512, H/8,  W/8)
        s4 = self.stage3(s3)       # (B, 1024,H/16, W/16)
        s5 = self.stage4(s4)       # (B, 2048,H/32, W/32)
        return [s1, s2, s3, s4, s5]


# ─── DenseNet-121 Encoder (Nhánh Perfusion — 12 kênh) ───────────────────────

class DenseNet121Encoder(nn.Module):
    """
    DenseNet-121 encoder trích xuất 5 cấp skip features cho UNet decoder.

    Skip output channels: [64, 128, 256, 512, 1024]
    Phù hợp với Perfusion map vì Dense Connection tái sử dụng feature hiệu quả,
    giúp giữ lại thông tin gradient màu sắc mờ nhạt của Tmax/CBF.
    """

    def __init__(self, in_channels: int = 12, weights_path: str = None):
        super().__init__()
        backbone = models.densenet121(weights=None)
        features = backbone.features

        # Inflate conv0: (64, 3, 7, 7) → (64, in_channels, 7, 7)
        old_conv0 = features.conv0
        new_conv0 = nn.Conv2d(
            in_channels,
            old_conv0.out_channels,
            kernel_size=old_conv0.kernel_size,
            stride=old_conv0.stride,
            padding=old_conv0.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv0.weight.copy_(
                inflate_weights(old_conv0.weight, in_channels)
            )
        features.conv0 = new_conv0

        # Load RadImageNet weights
        if weights_path is not None:
            self._load_radimagenet(backbone, weights_path, in_channels)

        # Tách 5 stage để lấy skip connections
        self.stage0 = nn.Sequential(
            features.conv0, features.norm0, features.relu0
        )                                          # 64ch, /2
        self.pool   = features.pool0               # /4 tổng
        self.stage1 = features.denseblock1         # 128ch (256 in original, half out)
        self.trans1 = features.transition1         # Compression → 128ch, /8
        self.stage2 = features.denseblock2         # 256ch
        self.trans2 = features.transition2         # → 256ch, /16
        self.stage3 = features.denseblock3         # 512ch
        self.trans3 = features.transition3         # → 512ch, /32
        self.stage4 = nn.Sequential(
            features.denseblock4, features.norm5
        )                                          # 1024ch, /32

    def _load_radimagenet(self, backbone: nn.Module, path: str, in_channels: int):
        state_dict = torch.load(path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Skip conv0 weight
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("features.conv0")}
        backbone.load_state_dict(state_dict, strict=False)
        print(f"[DenseNet121Encoder] Loaded RadImageNet weights from: {path}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Skip connections được lấy TRƯỚC Transition layers (trước khi downsample).
        Đảm bảo spatial size khớp với ResNet encoder ở mỗi level.

        Returns:
            List 5 feature maps từ nông → sâu:
            [d1(64,H/2), d2(256,H/4), d3(512,H/8), d4(1024,H/16), d5(1024,H/32)]
        """
        d1 = self.stage0(x)         # (B, 64,   H/2,  W/2)
        x  = self.pool(d1)          # (B, 64,   H/4,  W/4)

        d2 = self.stage1(x)         # (B, 256,  H/4,  W/4) ← skip trước trans1
        x  = self.trans1(d2)        # (B, 128,  H/8,  W/8)

        d3 = self.stage2(x)         # (B, 512,  H/8,  W/8) ← skip trước trans2
        x  = self.trans2(d3)        # (B, 256,  H/16, W/16)

        d4 = self.stage3(x)         # (B, 1024, H/16, W/16) ← skip trước trans3
        x  = self.trans3(d4)        # (B, 512,  H/32, W/32)

        d5 = self.stage4(x)         # (B, 1024, H/32, W/32)

        return [d1, d2, d3, d4, d5]



# ─── Factory ─────────────────────────────────────────────────────────────────

def build_encoders(config: dict):
    """
    Khởi tạo 2 encoder từ config dict (đọc từ model.yaml).

    Returns:
        (cta_encoder, perfusion_encoder)
    """
    cta_enc = ResNet50Encoder(
        in_channels=config["cta_encoder"]["in_channels"],
        weights_path=config["cta_encoder"]["weights"],
    )
    perf_enc = DenseNet121Encoder(
        in_channels=config["perfusion_encoder"]["in_channels"],
        weights_path=config["perfusion_encoder"]["weights"],
    )
    return cta_enc, perf_enc
