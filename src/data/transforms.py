import monai.transforms as mt
import numpy as np

def build_train_transforms(cfg: dict):
    """
    Train transforms with Spatial and Intensity Augmentations.
    Reads parameters from cfg['augmentation'].
    """
    keys = ["image", "label"]
    aug_cfg = cfg.get("augmentation", {})
    if not aug_cfg.get("enabled", True):
        return mt.Compose([mt.EnsureTyped(keys=keys, track_meta=False)])

    # Extract configs
    h_flip_prob = aug_cfg.get("horizontal_flip", {}).get("prob", 0.5)
    
    affine_cfg = aug_cfg.get("affine", {})
    affine_prob = affine_cfg.get("prob", 0.5)
    rot_range = affine_cfg.get("rotate_range", 0.26)
    s_range = affine_cfg.get("scale_range", 0.1)
    t_range = affine_cfg.get("translate_range", [10, 10])
    
    gn_cfg = aug_cfg.get("gaussian_noise", {})
    si_cfg = aug_cfg.get("scale_intensity", {})

    return mt.Compose([
        # Spatial Augmentations
        mt.RandHorizontalFlipd(keys=keys, prob=h_flip_prob),
        mt.RandAffined(
            keys=keys,
            prob=affine_prob,
            rotate_range=(rot_range, rot_range),
            translate_range=t_range,
            scale_range=(s_range, s_range),
            mode=("bilinear", "nearest"),
            padding_mode="zeros"
        ),
        
        # Random Crop & Resize
        mt.RandSpatialCropd(
            keys=keys,
            roi_size=(480, 480),
            random_center=True,
            random_size=False
        ),
        mt.Resized(keys=keys, spatial_size=(544, 544), mode=("bilinear", "nearest")),
        
        # Intensity Augmentations
        mt.RandGaussianNoised(
            keys=["image"],
            prob=gn_cfg.get("prob", 0.3),
            mean=gn_cfg.get("mean", 0.0),
            std=gn_cfg.get("std", 0.05)
        ),
        mt.RandScaleIntensityd(
            keys=["image"],
            factors=si_cfg.get("factors", 0.1),
            prob=si_cfg.get("prob", 0.3)
        ),
        
        # Ensure tensor
        mt.EnsureTyped(keys=keys, track_meta=False)
    ])

def build_val_transforms(cfg: dict):
    """
    Validation transforms. No augmentation.
    """
    keys = ["image", "label"]
    return mt.Compose([
        mt.EnsureTyped(keys=keys, track_meta=False)
    ])
