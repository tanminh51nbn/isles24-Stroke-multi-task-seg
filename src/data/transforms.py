"""
transforms.py — Augmentation pipeline cho ISLES'24

Nguyên tắc:
    - Spatial augmentation (flip, affine): áp dụng ĐỒNG BỘ lên cả input lẫn label
    - Intensity augmentation (noise, scale): CHỈ áp dụng lên input, KHÔNG áp dụng lên label
    - Label interpolation: NEAREST NEIGHBOR để giữ giá trị binary {0, 1}
    - Input interpolation: BILINEAR để giữ độ mượt của ảnh y tế
"""

import torch
import torch.nn as nn
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
    Lật trái-phải (Mirror Flip) ngẫu nhiên.
    
    Hợp lệ về mặt giải phẫu học vì:
    - Não người có tính đối xứng tương đối giữa 2 bán cầu
    - Đột quỵ xảy ra ngẫu nhiên cả hai bên trái và phải
    
    KHÔNG dùng lật trên-dưới (Anterior-Posterior) vì:
    - Hướng đầu trong nhữ́ng lát cắt axial là cố định
    - Lật dọc sẽ tạo ra ảnh không tồn tại trong thực tế
    
    flip(dims=[-1]) = lật theo chiều Width = lật trái-phải ✔
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


class RandomElasticTransform(nn.Module):
    """
    Biến dạng đàn hồi (Elastic Deformation) đã sửa lỗi.
    Mô phỏng nhu mô não bị chèn ép do sưng phù (edema).
    Sử dụng Separable Gaussian convolution để làm mịn displacement field trên GPU.
    """
    def __init__(self, prob: float = 0.4, alpha: float = 15.0, sigma: float = 6.0):
        super().__init__()
        self.prob  = prob
        self.alpha = alpha  # Độ lệch tối đa (pixel)
        self.sigma = sigma  # Độ mịn (pixel)

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.prob:
            return sample

        inp = sample["input"]   # (18, H, W)
        lbl = sample["label"]  # (3, H, W)
        C, H, W = inp.shape

        # 1. Tạo displacement field ngẫu nhiên thô
        dx = torch.randn(1, 1, H, W, device=inp.device)
        dy = torch.randn(1, 1, H, W, device=inp.device)

        # 2. Làm mượt field bằng Separable Gaussian convolution (nhanh hơn 2D conv)
        kernel_size = int(self.sigma * 4)
        if kernel_size % 2 == 0: kernel_size += 1
        
        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-x.pow(2) / (2 * self.sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_x = kernel_1d.view(1, 1, 1, -1).to(inp.device)
        kernel_y = kernel_1d.view(1, 1, -1, 1).to(inp.device)
        
        padding = kernel_size // 2
        dx = F.conv2d(F.conv2d(dx, kernel_x, padding=(0, padding)), kernel_y, padding=(padding, 0))
        dy = F.conv2d(F.conv2d(dy, kernel_x, padding=(0, padding)), kernel_y, padding=(padding, 0))
        
        # 3. Quy đổi độ lệch từ pixel sang normalized coordinate space [-1, 1]
        dx = dx * (self.alpha * 2.0 / W)
        dy = dy * (self.alpha * 2.0 / H)

        # 4. Tạo meshgrid chuẩn
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        grid = torch.stack([xx, yy], dim=-1).to(inp.device).unsqueeze(0) # (1, H, W, 2)

        # 5. Cộng displacement và áp dụng biến dạng
        disp = torch.cat([dx.permute(0, 2, 3, 1), dy.permute(0, 2, 3, 1)], dim=-1)
        grid = (grid + disp).clamp(-1, 1)

        sample["input"] = F.grid_sample(inp.unsqueeze(0), grid, mode="bilinear", align_corners=False).squeeze(0)
        sample["label"] = F.grid_sample(lbl.unsqueeze(0), grid, mode="nearest", align_corners=False).squeeze(0)
        return sample


class GaussianBlur:
    """Làm mờ Gaussian cho input (CHỈ input)."""
    def __init__(self, prob: float = 0.3, max_sigma: float = 1.0):
        self.prob = prob
        self.max_sigma = max_sigma

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.prob:
            return sample
        
        sigma = random.uniform(0.1, self.max_sigma)
        k_size = int(sigma * 4)
        if k_size % 2 == 0: k_size += 1
        
        inp = sample["input"] # (C, H, W)
        C, H, W = inp.shape
        
        # Tạo kernel 1D
        x = torch.arange(k_size).float() - (k_size - 1) / 2
        kernel_1d = torch.exp(-x.pow(2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Tạo kernel 2D
        kernel_2d = kernel_1d.view(1, 1, -1, 1) * kernel_1d.view(1, 1, 1, -1)
        kernel_2d = kernel_2d.expand(C, 1, k_size, k_size).to(inp.device)
        
        inp_blur = F.conv2d(inp.unsqueeze(0), kernel_2d, padding=k_size//2, groups=C)
        sample["input"] = inp_blur.squeeze(0)
        return sample


class RandomGamma:
    """Điều chỉnh Gamma ngẫu nhiên (CHỈ input)."""
    def __init__(self, prob: float = 0.5, gamma_range: tuple = (0.7, 1.3)):
        self.prob = prob
        self.gamma_range = gamma_range

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.prob:
            gamma = random.uniform(*self.gamma_range)
            inp = torch.clamp(sample["input"], min=0.0, max=1.0)
            sample["input"] = torch.pow(inp, gamma)
        return sample


class RandomModalityDropout:
    """
    Tắt ngẫu nhiên toàn bộ kênh CTA hoặc toàn bộ kênh Perfusion.
    Ép mô hình không được phụ thuộc vào 1 nhánh duy nhất, giúp chống overfitting mạnh.
    """
    def __init__(self, prob: float = 0.15):
        self.prob = prob
        # Index theo cấu hình channel_split của ISLES24
        self.cta_idx  = [0, 1, 6, 7, 12, 13]
        self.perf_idx = [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17]

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.prob:
            if random.random() < 0.5:
                # Tắt CTA (50% của prob)
                sample["input"][self.cta_idx, :, :] = 0.0
            else:
                # Tắt Perfusion (50% của prob)
                sample["input"][self.perf_idx, :, :] = 0.0
        return sample


class RandomChannelDropout:
    """
    Tắt ngẫu nhiên từ 1 đến max_drop kênh độc lập (ví dụ: mất Tmax ở Z-1, mất CBV ở Z).
    Mô phỏng nhiễu cục bộ và ép mô hình phải nội suy từ các kênh còn lại hoặc các lát cắt liền kề.
    """
    def __init__(self, prob: float = 0.2, max_drop: int = 3):
        self.prob = prob
        self.max_drop = max_drop

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.prob:
            num_channels = sample["input"].shape[0]  # Thường là 18
            n_drop = random.randint(1, self.max_drop)
            drop_indices = random.sample(range(num_channels), n_drop)
            sample["input"][drop_indices, :, :] = 0.0
        return sample


class RandomGridDistortion:
    """Biến dạng lưới (Grid Distortion) đồng bộ cho input và label."""
    def __init__(self, prob: float = 0.4, num_steps: int = 5, distort_limit: float = 0.2):
        self.prob = prob
        self.num_steps = num_steps
        self.distort_limit = distort_limit

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.prob:
            return sample

        inp = sample["input"]
        lbl = sample["label"]
        C, H, W = inp.shape

        # Tạo lưới biến dạng
        x_steps = torch.linspace(-1, 1, self.num_steps + 1)
        y_steps = torch.linspace(-1, 1, self.num_steps + 1)
        
        x_noise = (torch.rand(self.num_steps + 1) - 0.5) * self.distort_limit * (2.0 / self.num_steps)
        y_noise = (torch.rand(self.num_steps + 1) - 0.5) * self.distort_limit * (2.0 / self.num_steps)
        
        # Cố định các cạnh để tránh hở lưới
        x_noise[0] = x_noise[-1] = 0
        y_noise[0] = y_noise[-1] = 0
        
        x_distorted = (x_steps + x_noise).to(inp.device)
        y_distorted = (y_steps + y_noise).to(inp.device)

        # Nội suy lưới (Grid Interpolation)
        yy, xx = torch.meshgrid(torch.linspace(0, self.num_steps, H), torch.linspace(0, self.num_steps, W), indexing='ij')
        
        # Trích xuất phần nguyên và phần dư để nội suy tuyến tính
        x_idx = xx.long().clamp(0, self.num_steps - 1)
        y_idx = yy.long().clamp(0, self.num_steps - 1)
        x_frac = xx - x_idx.float()
        y_frac = yy - y_idx.float()

        # Nội suy tọa độ X và Y mới
        new_x = (1 - x_frac) * x_distorted[x_idx] + x_frac * x_distorted[x_idx + 1]
        new_y = (1 - y_frac) * y_distorted[y_idx] + y_frac * y_distorted[y_idx + 1]
        
        grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0).to(inp.device)

        sample["input"] = F.grid_sample(inp.unsqueeze(0), grid, mode="bilinear", align_corners=False).squeeze(0)
        sample["label"] = F.grid_sample(lbl.unsqueeze(0), grid, mode="nearest", align_corners=False).squeeze(0)
        return sample


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_train_transforms(config: dict) -> Callable:
    """Xây dựng augmentation pipeline cho tập train từ config."""
    aug = config["augmentation"]
    transforms = []

    if aug.get("enabled", True):
        # 1. Spatial Transforms (Sync Input + Label)
        if aug.get("horizontal_flip", {}).get("prob", 0) > 0:
            transforms.append(RandomHorizontalFlip(prob=aug["horizontal_flip"]["prob"]))
            
        transforms.append(RandomAffine(
            prob=aug["affine"]["prob"],
            rotate_deg=aug["affine"]["rotate_deg"],
            scale_range=aug["affine"]["scale_range"],
            translate_px=aug["affine"]["translate_px"],
        ))

        if "elastic_transform" in aug:
            transforms.append(RandomElasticTransform(
                prob=aug["elastic_transform"]["prob"],
                alpha=aug["elastic_transform"]["alpha"],
                sigma=aug["elastic_transform"]["sigma"]
            ))

        # 2. Intensity Transforms (Input Only)
        if "gaussian_blur" in aug:
            transforms.append(GaussianBlur(
                prob=aug["gaussian_blur"]["prob"],
                max_sigma=aug["gaussian_blur"]["max_sigma"]
            ))

        if "random_gamma" in aug:
            transforms.append(RandomGamma(
                prob=aug["random_gamma"]["prob"],
                gamma_range=aug["random_gamma"]["gamma_range"]
            ))

        if "grid_distortion" in aug:
            transforms.append(RandomGridDistortion(
                prob=aug["grid_distortion"]["prob"],
                num_steps=aug["grid_distortion"]["num_steps"],
                distort_limit=aug["grid_distortion"]["distort_limit"]
            ))

        transforms.append(GaussianNoise(
            prob=aug["gaussian_noise"]["prob"],
            std=aug["gaussian_noise"]["std"],
        ))
        transforms.append(RandomIntensityScale(
            prob=aug["intensity_scale"]["prob"],
            factor=aug["intensity_scale"]["factor"],
        ))
        
        # [Giải pháp B] Modality Dropout (Tắt toàn bộ nhánh)
        if "modality_dropout" in aug:
            transforms.append(RandomModalityDropout(
                prob=aug["modality_dropout"]["prob"]
            ))

        # [Giải pháp B mở rộng] Channel Dropout (Tắt ngẫu nhiên 1-3 kênh độc lập)
        if "channel_dropout" in aug:
            transforms.append(RandomChannelDropout(
                prob=aug["channel_dropout"]["prob"],
                max_drop=aug["channel_dropout"]["max_drop"]
            ))

    return Compose(transforms)


def build_val_transforms() -> Callable:
    """Validation: không augmentation, trả về nguyên sample."""
    return Compose([])
