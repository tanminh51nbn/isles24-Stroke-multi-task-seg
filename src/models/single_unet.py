"""
single_unet.py — Module: Single-Encoder Multi-Task UNet

Kiến trúc tổng thể:
    Input (18, 256, 256) (Tất cả CTA + Perfusion)
        ↓
    DenseNet-121 Encoder (Early Fusion) → [d1..d5]
        ↓
    Single Decoder (với các skip connections d1..d5)
        ↓
    MultiTaskHeads (Lesion, LVO, CoW)
"""

import torch
import torch.nn as nn
from typing import List, Optional

from models.encoder import build_encoders
from models.decoder import DecoderBlock, ConvBnGelu1x1
from models.heads import MultiTaskHeads


class SingleHeadDecoder(nn.Module):
    """
    Decoder chuẩn mực dành cho 1 Encoder duy nhất.
    Lấy cảm hứng từ kiến trúc decoder trong DualEncoderUNet nhưng gọt bỏ phần FeatureFusionBlock
    vì chúng ta chỉ có 1 luồng skip connections.
    """
    def __init__(self, config: dict, skip_channels: List[int]):
        super().__init__()
        
        dec_cfg = config["decoder"]
        out_ch = dec_cfg["out_channels"]  # vd: [256, 128, 64, 32]
        attention_type = dec_cfg.get("attention_type", None)
        
        # Bắt đầu từ stage sâu nhất (vd DenseNet121 stage4 ra 1024)
        in_ch = skip_channels[-1]  # 1024
        
        # Sẽ cần 4 block upsample tương ứng với 4 mức skip connection còn lại
        self.blocks = nn.ModuleList()
        for i in range(4):
            # Tính skip connection channel tương ứng (bỏ qua stage sâu nhất vì nó là input rồi)
            # Nếu skip_channels = [64, 256, 512, 1024, 1024]
            # i=0 (lên 512): skip = skip_channels[3] = 1024
            # i=1 (lên 256): skip = skip_channels[2] = 512
            # i=2 (lên 128): skip = skip_channels[1] = 256
            # i=3 (lên 64):  skip = skip_channels[0] = 64
            skip_idx = 3 - i
            skip_ch = skip_channels[skip_idx]
            
            # DecoderBlock gốc mong chờ cta_skip_ch và perf_skip_ch. 
            # Vì ta chỉ có 1 skip, ta truyền skip_ch vào cta_skip_ch, và 0 vào perf_skip_ch.
            # Nhưng FeatureFusionBlock cần Conv1x1. Truyền 0 kênh sẽ lỗi.
            # Giải pháp: Build khối Decoder đơn giản ngay tại đây.
            
            # ConvBnGelu1x1 kết hợp in_ch và skip_ch
            block = nn.ModuleDict({
                "upsample": nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                "conv_skip": ConvBnGelu1x1(skip_ch, skip_ch // 2),  # Giảm bớt skip channel
                "conv_mix": nn.Sequential(
                    nn.Conv2d(in_ch + (skip_ch // 2), out_ch[i], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch[i]),
                    nn.GELU(),
                    nn.Conv2d(out_ch[i], out_ch[i], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch[i]),
                    nn.GELU()
                )
            })
            self.blocks.append(block)
            in_ch = out_ch[i]

        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, dec_cfg.get("final_ch", 16), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_cfg.get("final_ch", 16)),
            nn.GELU()
        )

    def forward(self, skips: List[torch.Tensor], epoch: int = 0) -> tuple:
        """
        Args:
            skips: [stage0, stage1, stage2, stage3, stage4]
        """
        x = skips[-1] # Bắt đầu từ stage4
        
        aux_masks = []
        g_maps = []
        
        for i, block in enumerate(self.blocks):
            # Upsample current features
            x_up = block["upsample"](x)
            
            # Prepare skip features
            skip_idx = 3 - i
            s = skips[skip_idx]
            s = block["conv_skip"](s)
            
            # Concatenate & mix
            x = torch.cat([x_up, s], dim=1)
            x = block["conv_mix"](x)
            
        x = self.final_conv(x)
        
        # Hiện tại trả về rỗng cho aux_masks và g_maps để tương thích với output gốc
        return x, aux_masks, g_maps


class SingleEncoderUNet(nn.Module):
    """
    Single-Encoder Multi-Task UNet cho bài toán Stroke Segmentation.
    Sử dụng Early Fusion: nạp toàn bộ 18 kênh (CTA + Perfusion) vào layer đầu tiên.
    """

    def __init__(self, config: dict):
        super().__init__()
        
        # Sử dụng chung hàm build_encoders, nhưng ta chỉ lấy encoder chính
        # Vì model.yaml sẽ được cấu hình chỉ có "encoder" (hoặc truyền qua "cta_encoder" để tái sử dụng mã)
        
        if "encoder" in config:
            enc_cfg = config["encoder"]
        else:
            # Tương thích ngược nếu model.yaml vẫn cấu hình theo kiểu cũ
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

        # Decoder đơn nhánh
        self.decoder = SingleHeadDecoder(
            config,
            skip_channels=self.encoder.skip_channels
        )

        # 3 heads độc lập
        decoder_final_ch = config["decoder"].get("final_ch", 16)
        self.heads = MultiTaskHeads(
            in_ch=decoder_final_ch,
            dropout=config["heads"]["dropout"],
        )

    def forward(self, x: torch.Tensor, epoch: int = 0) -> dict:
        # Không tách kênh, nạp thẳng toàn bộ 18 kênh vào encoder
        skips = self.encoder(x)

        # Decode
        shared_features, aux_masks, g_maps = self.decoder(skips, epoch=epoch)

        # Cung cấp cùng 1 feature map cho cả 3 heads
        features_dict = {
            "lesion": shared_features,
            "lvo": shared_features,
            "cow": shared_features
        }

        # Heads
        out = self.heads(features_dict)
        
        # Mặc định aux_masks và g_maps rỗng
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
        """
        Trả về param groups cho AdamW với Differential LR:
            - Encoder: encoder_lr (thấp hơn để bảo vệ RadImageNet weights)
            - Decoder + Heads: decoder_lr (cao hơn để học nhanh)
        """
        return [
            {
                "params": list(self.encoder.parameters()),
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

def build_model(config: dict) -> nn.Module:
    """Tự động trả về mô hình tương ứng"""
    if "perfusion_encoder" in config:
        from models.dual_unet import DualEncoderUNet
        return DualEncoderUNet(config)
    else:
        return SingleEncoderUNet(config)
