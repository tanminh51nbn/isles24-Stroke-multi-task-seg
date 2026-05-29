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
        ├── TaskPath (Lesion, guidance=CoW)
        └── TaskPath (LVO, guidance=CoW+Lesion)
        ↓
    MultiTaskHeads (Lesion, LVO, CoW)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from models.encoder import build_encoders
from models.decoder import ConvBnGelu1x1, ConvBnGelu, LightweightDualAttention, AttentionGate, AuxHead
from models.heads import MultiTaskHeads


# ─── Khối Decoder cho 1 Encoder (Single Decoder Block) ─────────────────────────

class SingleDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention_type: Optional[str] = "dual", use_aux: bool = True, task_name: str = "shared", aux_ch: int = 1):
        super().__init__()
        self.attention_type = attention_type
        self.use_aux = use_aux
        self.aux_ch = aux_ch
        self.task_name = task_name
        
        self.conv1 = ConvBnGelu1x1(in_ch + skip_ch + aux_ch, out_ch)
        self.conv2 = ConvBnGelu(out_ch, out_ch)
        self.dropout = nn.Dropout2d(p=0.2)
        
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
        
        if self.attention_type == "ag":
            skip = self.ag(g=x_up, x=skip)
        elif self.attention_type == "dual":
            skip = self.dual_attn(skip)
            
        out = torch.cat([x_up, skip, prev_mask], dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.dropout(out)
        aux_out = self.aux_head(out) if self.use_aux else None
        return out, aux_out


# ─── Specialized Decoder Paths (Single Encoder version) ────────────────────────

class SingleSharedPath(nn.Module):
    def __init__(self, config: dict, skip_channels: List[int]):
        super().__init__()
        dec_ch = config["decoder"]["out_channels"]
        attn_type = config["decoder"].get("attention_type", "dual")

        # skip_channels = [s4, s3] (với s5 là bottleneck)
        self.dec4 = SingleDecoderBlock(1024, skip_channels[0], dec_ch[0], attn_type, use_aux=False, aux_ch=0)
        self.dec3 = SingleDecoderBlock(dec_ch[0], skip_channels[1], dec_ch[1], attn_type, use_aux=False, aux_ch=0)

    def forward(self, x_bottleneck, skips_shared):
        s4, s3 = skips_shared
        x, _ = self.dec4(x_bottleneck, s4, prev_mask=None)
        x, _ = self.dec3(x, s3, prev_mask=None)
        return x


class SingleTaskPath(nn.Module):
    def __init__(self, in_ch: int, config: dict, task_name: str, skip_channels: List[int], aux_ch: int = 1, active_aux_levels: List[bool] = [True, True], guidance_ch: int = 0):
        super().__init__()
        self.task_name = task_name
        dec_ch = config["decoder"]["out_channels"]
        final_ch = config["decoder"].get("final_ch", 16)
        attn_type = config["decoder"].get("attention_type", "dual")

        self.dec2 = SingleDecoderBlock(in_ch, skip_channels[0], dec_ch[2], attn_type, use_aux=active_aux_levels[0], task_name=task_name, aux_ch=aux_ch)
        self.dec1 = SingleDecoderBlock(dec_ch[2], skip_channels[1], dec_ch[3], attn_type, use_aux=active_aux_levels[1], task_name=task_name, aux_ch=aux_ch)

        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = ConvBnGelu(dec_ch[3], final_ch)
        
        if guidance_ch > 0:
            self.guidance_fusion = nn.Sequential(
                nn.Conv2d(final_ch + guidance_ch, final_ch, kernel_size=1),
                nn.BatchNorm2d(final_ch),
                nn.GELU()
            )
        else:
            self.guidance_fusion = None

    def forward(self, x_shared, skips_task, guidance: Optional[torch.Tensor] = None):
        s2, s1 = skips_task

        x, aux2 = self.dec2(x_shared, s2, prev_mask=None)
        x, aux1 = self.dec1(x, s1, prev_mask=aux2)

        x = self.up_final(x)
        x = self.final_conv(x)
        
        if guidance is not None and self.guidance_fusion is not None:
            g_interp = F.interpolate(guidance, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = self.guidance_fusion(torch.cat([x, g_interp], dim=1))

        return x, [None, None, aux2, aux1]


class SingleEncoderTripleDecoder(nn.Module):
    def __init__(self, config: dict, skip_channels: List[int]):
        super().__init__()
        bottleneck_ch = 1024
        
        # skip_channels = [ch_s1, ch_s2, ch_s3, ch_s4, ch_s5]
        
        self.shared_bottleneck = nn.Sequential(
            ConvBnGelu1x1(skip_channels[4], bottleneck_ch),
            ConvBnGelu(bottleneck_ch, bottleneck_ch),
        )

        skips_shared = [skip_channels[3], skip_channels[2]]  # s4, s3
        skips_task   = [skip_channels[1], skip_channels[0]]  # s2, s1
        
        self.shared_path = SingleSharedPath(config, skips_shared)
        
        dec_ch = config["decoder"]["out_channels"]
        in_ch_task = dec_ch[1] # output of dec3
        
        self.cow_path    = SingleTaskPath(in_ch_task, config, "cow", skips_task, aux_ch=1, active_aux_levels=[True, True], guidance_ch=0)
        self.lesion_path = SingleTaskPath(in_ch_task, config, "lesion", skips_task, aux_ch=1, active_aux_levels=[True, True], guidance_ch=16)
        self.lvo_path    = SingleTaskPath(in_ch_task, config, "lvo", skips_task, aux_ch=1, active_aux_levels=[True, True], guidance_ch=32)

    def forward(self, skips: List[torch.Tensor], epoch: int = 0):
        s1, s2, s3, s4, s5 = skips

        # 1. Bottleneck
        x_bottleneck = self.shared_bottleneck(s5)
        
        # 2. Shared Path (dec4, dec3)
        x_shared = self.shared_path(x_bottleneck, [s4, s3])
        
        f_cow, cow_auxs = self.cow_path(x_shared, [s2, s1])
        
        f_lesion, lesion_auxs = self.lesion_path(x_shared, [s2, s1], guidance=f_cow.detach())
        
        guidance_for_lvo = torch.cat([f_cow.detach(), f_lesion.detach()], dim=1)
        if self.training:
            guidance_for_lvo.requires_grad_(True)
            
            def lvo_guidance_hook(grad):
                # Ghi lại norm của gradient chảy qua guidance của LVO
                norm = grad.norm(2).item()
                # Lưu vào thuộc tính tạm để in ra ở cấp trainer nếu cần
                self._lvo_guidance_grad_norm = norm
                
            guidance_for_lvo.register_hook(lvo_guidance_hook)
            
        f_lvo, lvo_auxs = self.lvo_path(x_shared, [s2, s1], guidance=guidance_for_lvo)

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
            dropout=config["heads"]["dropout"],
        )

    def forward(self, x: torch.Tensor, epoch: int = 0) -> dict:
        skips = self.encoder(x)

        features_dict, aux_masks, g_maps = self.decoder(skips, epoch=epoch)

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
        model = DualEncoderUNet(config)
        model_name = "DualEncoderUNet"
    else:
        model = SingleEncoderUNet(config)
        model_name = "SingleEncoderUNet"
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Total params: {total_params:,} | Trainable: {trainable:,}")
    return model
