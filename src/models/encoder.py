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
    NÂNG CẤP: Thổi phồng weight với trọng số tập trung vào tâm (Center-Weighted).
    Giả định lát cắt ở giữa chứa thông tin quan trọng nhất, các lát cắt rìa là bối cảnh.
    """
    out_ch, in_ch, kH, kW = weight.shape # in_ch thường là 3
    
    # 1. Lấy đặc trưng trung bình từ 3 kênh gốc
    mean_weight = weight.mean(dim=1, keepdim=True) # (out_ch, 1, kH, kW)
    
    # 2. Tạo profile trọng số cho các lát cắt (Gaussian-like)
    # Ví dụ với 9 kênh: [0.5, 0.7, 0.9, 1.0, 1.1, 1.0, 0.9, 0.7, 0.5]
    center = target_channels // 2
    indices = torch.arange(target_channels).float()
    # Tính khoảng cách tới tâm, càng xa tâm trọng số càng giảm nhẹ
    weights = torch.exp(-0.1 * (indices - center)**2)
    weights = weights / weights.sum() * 3.0 # Chuẩn hóa để tổng năng lượng tương đương 3 kênh gốc
    
    # 3. Thổi phồng và áp trọng số
    new_weight = mean_weight.repeat(1, target_channels, 1, 1)
    for i in range(target_channels):
        new_weight[:, i, :, :] *= weights[i]
        
    return new_weight


class SliceAttention(nn.Module):
    """
    Learnable Slice Attention (Dual Pooling): 
    Kết hợp Global Average Pooling (Context) và Global Max Pooling (Saliency) 
    để mô hình tự học cách nhấn mạnh các lát cắt quan trọng.
    Sử dụng cơ chế Concat thay vì Add để bảo toàn đặc trưng đa dạng.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Branch 1: Avg (Contextual info)
        avg_out = self.avg_pool(x).view(b, c)
        
        # Branch 2: Max (Saliency/LVO info)
        max_out = self.max_pool(x).view(b, c)
        
        # Combine via Concatenation
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.fc(combined)
        
        out = self.sigmoid(out).view(b, c, 1, 1)
        
        return x * out.expand_as(x)


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

        # 1. Slice Attention để học trọng số lát cắt
        self.slice_attention = SliceAttention(in_channels)

        # 2. Backbone Setup
        # Chúng ta khởi tạo Conv1 với số kênh mong muốn, trọng số sẽ được nạp sau
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1   = backbone.bn1
        self.gelu  = nn.GELU()

        # Load RadImageNet weights (nếu có)
        if weights_path is not None:
            self._load_radimagenet(weights_path, in_channels)
        else:
            # Fallback: Inflate từ ImageNet nếu không có RadImageNet path
            with torch.no_grad():
                self.conv1.weight.copy_(
                    inflate_weights(backbone.conv1.weight, in_channels)
                )

        # Tách thành 5 stage để lấy skip connections
        self.stage0 = nn.Sequential(self.slice_attention, self.conv1, self.bn1, self.gelu)  # 64ch, /2
        self.pool   = backbone.maxpool  # /4 tổng
        self.stage1 = backbone.layer1   # 256ch,  /4
        self.stage2 = backbone.layer2   # 512ch,  /8
        self.stage3 = backbone.layer3   # 1024ch, /16
        self.stage4 = backbone.layer4   # 2048ch, /32

    def _load_radimagenet(self, path: str, in_channels: int):
        """Load RadImageNet checkpoint, thực hiện inflate conv1 từ trọng số y tế."""
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        
        # Xử lý wrapper
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # 1. Trích xuất và Inflate Conv1 từ RadImageNet (Chuẩn y tế)
        if "conv1.weight" in state_dict:
            rad_conv1_weight = state_dict["conv1.weight"]
            with torch.no_grad():
                self.conv1.weight.copy_(
                    inflate_weights(rad_conv1_weight, in_channels)
                )
            # Xóa conv1 khỏi state_dict để không gây lỗi size mismatch khi load_state_dict cho phần còn lại
            del state_dict["conv1.weight"]

        # 2. Load các layer còn lại (layer1-4, bn1, v.v.) vào chính model này
        # Lưu ý: Chúng ta load vào 'self' thay vì 'backbone' vì các thuộc tính đã được gán sang self
        msg = self.load_state_dict(state_dict, strict=False)
        print(f"[ResNet50Encoder] Pure RadImageNet Loaded. Missing keys: {len(msg.missing_keys)}")

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
        # 1. Slice Attention để học trọng số lát cắt
        self.slice_attention = SliceAttention(in_channels)

        # 2. Backbone Setup
        # DenseNet dùng features.conv0 thay vì conv1
        self.conv0 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm0 = features.norm0
        self.gelu0 = nn.GELU()

        # Load RadImageNet weights
        if weights_path is not None:
            self._load_radimagenet(weights_path, in_channels)
        else:
            # Fallback: Inflate từ ImageNet
            with torch.no_grad():
                self.conv0.weight.copy_(
                    inflate_weights(features.conv0.weight, in_channels)
                )

        # Tách 5 stage để lấy skip connections
        self.stage0 = nn.Sequential(
            self.slice_attention, self.conv0, self.norm0, self.gelu0
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

    def _load_radimagenet(self, path: str, in_channels: int):
        """Load RadImageNet cho DenseNet, thực hiện inflate conv0 từ trọng số y tế."""
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # 1. Inflate Conv0 (features.conv0.weight trong RadImageNet)
        if "features.conv0.weight" in state_dict:
            rad_conv0_weight = state_dict["features.conv0.weight"]
            with torch.no_grad():
                self.conv0.weight.copy_(
                    inflate_weights(rad_conv0_weight, in_channels)
                )
            del state_dict["features.conv0.weight"]

        # 2. Load các layer còn lại
        msg = self.load_state_dict(state_dict, strict=False)
        print(f"[DenseNet121Encoder] Pure RadImageNet Loaded. Missing: {len(msg.missing_keys)}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Skip connections được lấy TRƯỚC Transition layers (trước khi downsample).
        Đảm bảo spatial size khớp với ResNet encoder ở mỗi level.

        Returns:
            List 5 feature maps từ nông → sâu:
            [d1(64,H/2), d2(128,H/4), d3(256,H/8), d4(512,H/16), d5(1024,H/32)]
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
    cta_name = config["cta_encoder"].get("name", "resnet50").lower()
    if cta_name == "densenet121":
        cta_enc = DenseNet121Encoder(
            in_channels=config["cta_encoder"]["in_channels"],
            weights_path=config["cta_encoder"]["weights"],
        )
    else:
        cta_enc = ResNet50Encoder(
            in_channels=config["cta_encoder"]["in_channels"],
            weights_path=config["cta_encoder"]["weights"],
        )

    perf_name = config["perfusion_encoder"].get("name", "densenet121").lower()
    if perf_name == "densenet121":
        perf_enc = DenseNet121Encoder(
            in_channels=config["perfusion_encoder"]["in_channels"],
            weights_path=config["perfusion_encoder"]["weights"],
        )
    else:
        # Fallback to ResNet50 if somehow specified
        perf_enc = ResNet50Encoder(
            in_channels=config["perfusion_encoder"]["in_channels"],
            weights_path=config["perfusion_encoder"]["weights"],
        )
        
    return cta_enc, perf_enc
