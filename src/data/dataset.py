import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import binary_dilation, gaussian_filter

logger = logging.getLogger(__name__)

def create_disk_kernel(radius):
    """Tạo nhân hình tròn cho phép giãn nhãn."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(bool)

class ISLES24Dataset(Dataset):
    """
    Dataset for ISLES'24 2.5D Multimodal Slices.
    Handles loading, negative downsampling, outlier clipping, and brain masking.
    """
    
    def __init__(
        self,
        patient_dirs: List[Path],
        transform=None,
        is_train: bool = False,
        downsample_neg_ratio: float = 1.0,
        lvo_oversample: int = 1,
        lvo_dilation_radius: int = 0
    ):
        self.patient_dirs = patient_dirs
        self.transform = transform
        self.is_train = is_train
        self.downsample_neg_ratio = downsample_neg_ratio
        self.lvo_oversample = lvo_oversample
        self.lvo_dilation_radius = lvo_dilation_radius
        
        self.slices = []
        self._build_index()
        
    def _build_index(self):
        """Index all available slices and apply sampling strategies if training."""
        raw_slices = []
        for pdir in self.patient_dirs:
            if not pdir.is_dir():
                continue
                
            img_dir = pdir / "inputs"
            lbl_dir = pdir / "labels"
            
            if not img_dir.exists() or not lbl_dir.exists():
                continue
                
            slice_files = sorted(list(img_dir.glob("x_z*.npy")))
            for sf in slice_files:
                lf = lbl_dir / sf.name.replace("x_z", "y_z")
                if lf.exists():
                    raw_slices.append({"image": sf, "label": lf})
                    
        if not self.is_train:
            self.slices = raw_slices
            logger.info(f"Validation Dataset: {len(self.slices)} slices loaded.")
            return
            
        # 4 Pools for Task-Balanced Sampling
        self.task_pools = {
            "lvo": [],
            "lesion": [],
            "cow": [],
            "neg": []
        }
        
        for i, item in enumerate(raw_slices):
            lbl = np.load(item["label"], mmap_mode='r')
            
            has_lesion = lbl[0].max() > 0
            has_lvo = lbl[1].max() > 0
            has_cow = lbl[2].max() > 0
            
            # Ưu tiên LVO > Lesion > CoW > Neg để đảm bảo tính duy nhất trong pool
            if has_lvo:
                self.task_pools["lvo"].append(i)
            elif has_lesion:
                self.task_pools["lesion"].append(i)
            elif has_cow:
                self.task_pools["cow"].append(i)
            else:
                self.task_pools["neg"].append(i)
                
        self.slices = raw_slices
        
        if self.is_train:
            logger.info(f"Dataset Pools: LVO={len(self.task_pools['lvo'])}, Lesion={len(self.task_pools['lesion'])}, CoW={len(self.task_pools['cow'])}, Neg={len(self.task_pools['neg'])}")

    def __len__(self):
        return len(self.slices)
    
    def get_task_indices(self):
        """Trả về các kho chứa chỉ mục cho Sampler."""
        return self.task_pools

    def __getitem__(self, idx):
        item = self.slices[idx]
        
        # Load data
        img = np.load(item["image"]).astype(np.float32) # [18, 544, 544]
        lbl = np.load(item["label"]).astype(np.float32) # [3, 544, 544]
        
        # 0. LVO Label Softening (Hard Core - Soft Shell)
        # Bán kính 5px tương đương sigma=2.0 để tạo quầng mờ mượt mà
        if self.lvo_dilation_radius > 0:
            lvo_mask = lbl[1] > 0
            if lvo_mask.any():
                # Tạo quầng mờ Gaussian
                sigma = self.lvo_dilation_radius / 2.5 # Rule of thumb for radius mapping
                soft = gaussian_filter(lvo_mask.astype(np.float32), sigma=sigma)
                if soft.max() > 0:
                    soft = soft / soft.max()
                
                # Kết hợp: Lõi nhãn thật = 1.0, Xung quanh mờ dần
                lbl[1] = np.maximum(lvo_mask.astype(np.float32), soft)
        
        # 1. Clip Perfusion Outliers (Channels 6 to 17) to [-5, 5]
        # Keep as a safety measure for extreme z-scores in already normalized data
        img[6:18] = np.clip(img[6:18], -5.0, 5.0)
        
        # Prepare dict for MONAI
        data = {"image": img, "label": lbl}
        
        if self.transform:
            data = self.transform(data)
            
        img_t = data["image"]
        lbl_t = data["label"]
        
        # Convert to tensor first to use torch methods
        if isinstance(img_t, np.ndarray):
            img_t = torch.from_numpy(img_t)
        if isinstance(lbl_t, np.ndarray):
            lbl_t = torch.from_numpy(lbl_t)
        
        # Ensure brain mask matches potential spatial augmentations
        final_brain_mask = (img_t[1:2] > -0.95).float()
        
        return img_t, lbl_t, final_brain_mask

def build_dataset(patient_dirs: List[Path], cfg: dict, is_train: bool = False, transform=None) -> ISLES24Dataset:
    """
    Factory function to build ISLES24Dataset from config.
    """
    samp_cfg = cfg.get("sampling", {})
    
    # Chỉ áp dụng sampling ratio (downsample/oversample) khi huấn luyện
    downsample_neg = samp_cfg.get("downsample_neg_ratio", 1.0) if is_train else 1.0
    lvo_over = samp_cfg.get("lvo_oversample", 1) if is_train else 1
    lvo_dilate = samp_cfg.get("lvo_dilation_radius", 0)
    
    return ISLES24Dataset(
        patient_dirs=patient_dirs,
        transform=transform,
        is_train=is_train,
        downsample_neg_ratio=downsample_neg,
        lvo_oversample=lvo_over,
        lvo_dilation_radius=lvo_dilate
    )
