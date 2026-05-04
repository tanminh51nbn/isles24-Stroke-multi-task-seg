"""
encoder.py — Generic Timm-based Dual-Encoder: ResNet-34 (CTA) + EfficientNet-B0 (Perfusion)

Nhiệm vụ:
    - Load ImageNet pre-trained weights cho từng backbone qua thư viện `timm`.
    - Inflate lớp conv đầu tiên từ 3 kênh lên N kênh (Conv1 Inflation)
    - Trả về feature extractor với 5 cấp skip connections
"""

import torch
import torch.nn as nn
import timm
from typing import List

# ─── Utility: Conv1 Inflation ────────────────────────────────────────────────

def inflate_weights(weight: torch.Tensor, target_channels: int) -> torch.Tensor:
    """
    NÂNG CẤP: Thổi phồng weight với trọng số tập trung vào tâm (Center-Weighted).
    """
    out_ch, in_ch, kH, kW = weight.shape # in_ch thường là 3
    
    mean_weight = weight.mean(dim=1, keepdim=True) 
    
    center = target_channels // 2
    indices = torch.arange(target_channels).float()
    weights = torch.exp(-0.1 * (indices - center)**2)
    weights = weights / weights.sum() * 3.0 
    
    new_weight = mean_weight.repeat(1, target_channels, 1, 1)
    for i in range(target_channels):
        new_weight[:, i, :, :] *= weights[i]
        
    return new_weight


class SliceAttention(nn.Module):
    """
    Learnable Slice Attention (Dual Pooling)
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2 + 1),
            nn.GELU(),
            nn.Linear(in_channels // 2 + 1, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


# ─── Generic Encoder using `timm` ──────────────────────────────────────────

class TimmEncoder(nn.Module):
    """
    Encoder dùng chung dựa trên thư viện `timm`.
    """
    def __init__(self, name: str, in_channels: int, weights_path: str = None):
        super().__init__()
        self.slice_attention = SliceAttention(in_channels)
        
        # Khởi tạo backbone từ timm, trả về các skip levels
        self.backbone = timm.create_model(
            name, 
            pretrained=False,
            features_only=True
        )
        
        first_conv_name = None
        first_conv_module = None
        
        for n, m in self.backbone.named_modules():
            if isinstance(m, nn.Conv2d):
                first_conv_name = n
                first_conv_module = m
                break
                
        if first_conv_module is None:
            raise ValueError(f"Không tìm thấy lớp Conv2d nào trong backbone {name}!")

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv_module.out_channels,
            kernel_size=first_conv_module.kernel_size,
            stride=first_conv_module.stride,
            padding=first_conv_module.padding,
            bias=(first_conv_module.bias is not None),
            groups=first_conv_module.groups
        )
        
        parts = first_conv_name.split('.')
        parent = self.backbone
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)

        self.first_conv = new_conv

        print(f"[TimmEncoder] Khởi tạo {name} (ImageNet pre-trained) với in_channels={in_channels}.")
        tmp_model = timm.create_model(name, pretrained=True)
        for n, m in tmp_model.named_modules():
            if isinstance(m, nn.Conv2d):
                old_weight = m.weight.data
                with torch.no_grad():
                    self.first_conv.weight.copy_(inflate_weights(old_weight, in_channels))
                break
                
        state_dict = tmp_model.state_dict()
        state_dict_filtered = {k: v for k, v in state_dict.items() if not k.startswith(first_conv_name + ".weight")}
        self.backbone.load_state_dict(state_dict_filtered, strict=False)
        del tmp_model

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.slice_attention(x)
        return self.backbone(x)


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_encoders(config: dict):
    """
    Khởi tạo 2 encoder từ config dict (đọc từ model.yaml).
    """
    cta_enc = TimmEncoder(
        name=config["cta_encoder"]["name"],
        in_channels=config["cta_encoder"]["in_channels"],
        weights_path=None,
    )
    perf_enc = TimmEncoder(
        name=config["perfusion_encoder"]["name"],
        in_channels=config["perfusion_encoder"]["in_channels"],
        weights_path=None,
    )
    return cta_enc, perf_enc

