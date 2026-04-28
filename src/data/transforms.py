"""
transforms.py — Augmentation pipeline cho ISLES'24

Nguyên tắc:
    - Spatial augmentation (flip, affine): áp dụng ĐỒNG BỘ lên cả input lẫn label
    - Intensity augmentation (noise, scale): CHỈ áp dụng lên input, KHÔNG áp dụng lên label
    - Label interpolation: NEAREST NEIGHBOR để giữ giá trị binary {0, 1}
    - Input interpolation: BILINEAR để giữ độ mượt của ảnh y tế
"""

import torch
import numpy as np
import random
from typing import Callable
import torch.nn.functional as F


class Compose:
    """Kết hợp nhiều transform theo thứ tự."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomHorizontalFlip:
    """
    Lật ngang ngẫu nhiên.
    Não đối xứng theo trục dọc → lật ngang là augmentation hoàn toàn hợp lệ.
    """
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.prob:
            sample["input"] = torch.flip(sample["input"], dims=[-1])
            sample["label"] = torch.flip(sample["label"], dims=[-1])
        return sample


class RandomAffine:
    """
    Biến đổi Affine ngẫu nhiên: xoay + scale + dịch chuyển.
    Áp dụng đồng bộ lên cả input và label (với interpolation khác nhau).
    """
    def __init__(
        self,
        prob: float = 0.5,
        rotate_deg: float = 15.0,
        scale_range: float = 0.1,
        translate_px: int = 10,
    ):
        self.prob         = prob
        self.rotate_deg   = rotate_deg
        self.scale_range  = scale_range
        self.translate_px = translate_px

    def _get_affine_matrix(self, H: int, W: int, device) -> torch.Tensor:
        """Tạo affine matrix ngẫu nhiên."""
        angle  = random.uniform(-self.rotate_deg, self.rotate_deg)
        scale  = random.uniform(1 - self.scale_range, 1 + self.scale_range)
        tx     = random.uniform(-self.translate_px, self.translate_px) / W * 2
        ty     = random.uniform(-self.translate_px, self.translate_px) / H * 2

        angle_rad = angle * (3.14159265 / 180.0)
        cos_a, sin_a = scale * np.cos(angle_rad), scale * np.sin(angle_rad)

        # Affine matrix 2×3
        theta = torch.tensor([
            [cos_a, -sin_a, tx],
            [sin_a,  cos_a, ty],
        ], dtype=torch.float32, device=device)
        return theta

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.prob:
            return sample

        inp = sample["input"]   # (18, H, W)
        lbl = sample["label"]  # (3, H, W)
        _, H, W = inp.shape

        theta = self._get_affine_matrix(H, W, inp.device)
        theta = theta.unsqueeze(0)  # (1, 2, 3)

        # Áp dụng cho input (bilinear)
        inp4d  = inp.unsqueeze(0)   # (1, 18, H, W)
        grid   = F.affine_grid(theta, inp4d.shape, align_corners=False)
        inp4d  = F.grid_sample(inp4d, grid, mode="bilinear", align_corners=False, padding_mode="zeros")

        # Áp dụng cho label (nearest — giữ giá trị {0, 1})
        lbl4d  = lbl.unsqueeze(0)   # (1, 3, H, W)
        lbl4d  = F.grid_sample(lbl4d, grid, mode="nearest", align_corners=False, padding_mode="zeros")

        sample["input"] = inp4d.squeeze(0)
        sample["label"] = lbl4d.squeeze(0)
        return sample


class GaussianNoise:
    """Thêm nhiễu Gaussian vào input (CHỈ input, KHÔNG label)."""
    def __init__(self, prob: float = 0.3, mean: float = 0.0, std: float = 0.05):
        self.prob = prob
        self.mean = mean
        self.std  = std

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.prob:
            noise = torch.randn_like(sample["input"]) * self.std + self.mean
            sample["input"] = torch.clamp(sample["input"] + noise, 0.0, 1.0)
        return sample


class RandomIntensityScale:
    """Scale độ sáng ngẫu nhiên (CHỈ input)."""
    def __init__(self, prob: float = 0.3, factor: float = 0.1):
        self.prob   = prob
        self.factor = factor

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.prob:
            scale = random.uniform(1 - self.factor, 1 + self.factor)
            sample["input"] = torch.clamp(sample["input"] * scale, 0.0, 1.0)
        return sample


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_train_transforms(config: dict) -> Callable:
    """Xây dựng augmentation pipeline cho tập train từ config."""
    aug = config["augmentation"]
    transforms = []

    if aug.get("enabled", True):
        transforms.append(RandomHorizontalFlip(
            prob=aug["horizontal_flip"]["prob"]
        ))
        transforms.append(RandomAffine(
            prob=aug["affine"]["prob"],
            rotate_deg=aug["affine"]["rotate_deg"],
            scale_range=aug["affine"]["scale_range"],
            translate_px=aug["affine"]["translate_px"],
        ))
        transforms.append(GaussianNoise(
            prob=aug["gaussian_noise"]["prob"],
            std=aug["gaussian_noise"]["std"],
        ))
        transforms.append(RandomIntensityScale(
            prob=aug["intensity_scale"]["prob"],
            factor=aug["intensity_scale"]["factor"],
        ))

    return Compose(transforms)


def build_val_transforms() -> Callable:
    """Validation: không augmentation, trả về nguyên sample."""
    return Compose([])
