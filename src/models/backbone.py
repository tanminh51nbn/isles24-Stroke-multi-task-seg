import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_rad_weights(encoder: nn.Module, rad_weight_path: str) -> None:
    """
    Load RadImageNet pretrained weights vào encoder.
    Hỗ trợ các format key phổ biến: raw, state_dict, model.encoder.*, encoder.*
    """
    if rad_weight_path is None:
        return

    try:
        logger.info(f"Loading RadImageNet weights from: {rad_weight_path}")
        rad_state_dict = torch.load(rad_weight_path, map_location="cpu")

        if "state_dict" in rad_state_dict:
            rad_state_dict = rad_state_dict["state_dict"]

        cleaned_dict = {}
        for k, v in rad_state_dict.items():
            if k.startswith("model.encoder."):
                cleaned_dict[k.replace("model.encoder.", "")] = v
            elif k.startswith("encoder."):
                cleaned_dict[k.replace("encoder.", "")] = v
            else:
                cleaned_dict[k] = v

        encoder.load_state_dict(cleaned_dict, strict=False)
        logger.info("Successfully loaded RadImageNet weights into encoder.")

    except Exception as e:
        logger.error(f"Failed to load RadImageNet weights: {e}")
        logger.warning("Proceeding with random initialization for encoder.")


def inflate_conv1(encoder: nn.Module, in_channels: int) -> None:
    """
    Inflate encoder conv1 từ 3 channels lên in_channels bằng Average-Repeat.

    Formula: new_weight[c] = mean(old_weight) / (in_channels / 3)
    Giữ tổng activation scale bất biến khi số channel tăng từ 3 lên in_channels.
    """
    if in_channels == 3:
        return

    old_conv = encoder.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )

    with torch.no_grad():
        # Average 3 pretrained channels → repeat in_channels lần → scale down
        mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.data = mean_weight.repeat(1, in_channels, 1, 1) / (in_channels / 3.0)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()

    encoder.conv1 = new_conv
    logger.info(f"Inflated conv1 from 3 to {in_channels} channels via Average-Repeat.")
