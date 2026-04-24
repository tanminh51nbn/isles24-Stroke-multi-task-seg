import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

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
        lvo_oversample: int = 1
    ):
        self.patient_dirs = patient_dirs
        self.transform = transform
        self.is_train = is_train
        self.downsample_neg_ratio = downsample_neg_ratio
        self.lvo_oversample = lvo_oversample
        
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
            
        # Training sampling logic
        final_slices = []
        rng = np.random.default_rng(seed=42)
        stats = {"total": 0, "neg": 0, "pos": 0, "lvo": 0, "cow": 0, "les": 0}
        
        for item in raw_slices:
            stats["total"] += 1
            
            # Fast check label presence without loading entire array if possible
            # But for simplicity and robustness, we load it here to check if it's positive.
            # In a real scenario with huge dataset, we should cache metadata.
            # Assuming labels are small enough to load quickly during init.
            lbl = np.load(item["label"], mmap_mode='r') # [3, 544, 544]
            
            has_lesion = lbl[0].max() > 0
            has_lvo = lbl[1].max() > 0
            has_cow = lbl[2].max() > 0
            
            is_positive = has_lesion or has_lvo or has_cow
            
            if is_positive:
                stats["pos"] += 1
                if has_lesion: stats["les"] += 1
                if has_lvo: stats["lvo"] += 1
                if has_cow: stats["cow"] += 1
                
                # Oversample LVO
                repeats = self.lvo_oversample if has_lvo else 1
                for _ in range(repeats):
                    final_slices.append(dict(item))
            else:
                stats["neg"] += 1
                # Downsample negative
                if rng.random() <= self.downsample_neg_ratio:
                    final_slices.append(item)
                    
        self.slices = final_slices
        logger.info(f"Training Dataset Stats (Raw): {stats}")
        logger.info(f"Training Dataset Final Size: {len(self.slices)} slices (Neg ratio: {self.downsample_neg_ratio}, LVO oversample: {self.lvo_oversample})")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        item = self.slices[idx]
        
        # Load data
        img = np.load(item["image"]).astype(np.float32) # [18, 544, 544]
        lbl = np.load(item["label"]).astype(np.float32) # [3, 544, 544]
        
        # 1. Clip Perfusion Outliers (Channels 6 to 17) to [-5, 5]
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
    
    return ISLES24Dataset(
        patient_dirs=patient_dirs,
        transform=transform,
        is_train=is_train,
        downsample_neg_ratio=downsample_neg,
        lvo_oversample=lvo_over
    )
