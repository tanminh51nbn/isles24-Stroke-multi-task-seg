"""
transforms.py — MONAI augmentation pipelines for ISLES'24 2.5D medical imaging.

Design principles:
  - ONLY spatial transforms (no intensity/color augmentation)
  - Image uses bilinear interpolation, label uses nearest (preserves binary)
  - CoarseDropout applies to image ONLY (never mask the ground truth)
  - No vertical flip (anterior ≠ posterior in axial CT)

Input format: {"image": [18, 544, 544], "label": [3, 544, 544]}
"""
from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAffined,
    RandCoarseDropoutd,
    RandFlipd,
    Rand2DElasticd,
)


def build_train_transforms(cfg: dict) -> Compose:
    """
    Build MONAI training transform pipeline from augmentation config.

    Pipeline order:
      1. HorizontalFlip  — left-right mirror (anatomically valid for brain)
      2. RandAffine      — rotation ±15°, scale ±10%, translate ±27px
      3. Elastic2D       — light deformation simulating anatomical variance
      4. CoarseDropout   — CutOut regularization (image only)
      5. EnsureType      — guarantee torch.Tensor float32 output

    Args:
        cfg: Full data config dict (uses cfg["augmentation"] section)

    Returns:
        MONAI Compose pipeline
    """
    aug_cfg = cfg.get("augmentation", {})

    if not aug_cfg.get("enabled", True):
        return build_val_transforms()

    transforms = []

    # ── 1. Horizontal Flip ──────────────────────────────────────
    # spatial_axis=1 → flip along W dimension in (C, H, W)
    # = left-right mirror, anatomically acceptable for brain CT
    flip_cfg = aug_cfg.get("horizontal_flip", {})
    if flip_cfg.get("prob", 0) > 0:
        transforms.append(
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=1,
                prob=flip_cfg["prob"],
            )
        )

    # ── 2. Random Affine ────────────────────────────────────────
    # Rotation + Scale + Translation — simulates patient positioning variance
    affine_cfg = aug_cfg.get("affine", {})
    if affine_cfg.get("prob", 0) > 0:
        rotate = affine_cfg.get("rotate_range", 0.26)
        scale = affine_cfg.get("scale_range", 0.1)
        translate = affine_cfg.get("translate_range", [27, 27])

        transforms.append(
            RandAffined(
                keys=["image", "label"],
                prob=affine_cfg["prob"],
                rotate_range=[rotate],              # Single angle for 2D plane
                scale_range=[scale, scale],          # ±scale for (H, W)
                translate_range=translate,            # ±px for (H, W)
                mode=("bilinear", "nearest"),        # image=bilinear, label=nearest
                padding_mode="zeros",
            )
        )

    # ── 3. Elastic Deformation 2D ───────────────────────────────
    # Light deformation — simulates inter-patient anatomical variance
    # Kept mild (magnitude 1-3) because brain anatomy is relatively rigid
    elastic_cfg = aug_cfg.get("elastic_2d", {})
    if elastic_cfg.get("prob", 0) > 0:
        transforms.append(
            Rand2DElasticd(
                keys=["image", "label"],
                spacing=tuple(elastic_cfg.get("spacing", [50, 50])),
                magnitude_range=tuple(elastic_cfg.get("magnitude_range", [1, 3])),
                prob=elastic_cfg["prob"],
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            )
        )

    # ── 4. Coarse Dropout (CutOut) ──────────────────────────────
    # Applied to IMAGE ONLY — forces model to use surrounding context
    # Never applied to label (would corrupt ground truth)
    dropout_cfg = aug_cfg.get("coarse_dropout", {})
    if dropout_cfg.get("prob", 0) > 0:
        transforms.append(
            RandCoarseDropoutd(
                keys=["image"],                      # ⚠️ Image only!
                holes=dropout_cfg.get("holes", 8),
                spatial_size=tuple(dropout_cfg.get("spatial_size", [32, 32])),
                fill_value=dropout_cfg.get("fill_value", 0.0),
                prob=dropout_cfg["prob"],
            )
        )

    # ── 5. Ensure Tensor Output ─────────────────────────────────
    transforms.append(
        EnsureTyped(keys=["image", "label"], dtype="float32")
    )

    return Compose(transforms)


def build_val_transforms() -> Compose:
    """
    Validation/Test pipeline: no augmentation, only ensure tensor type.
    """
    return Compose([
        EnsureTyped(keys=["image", "label"], dtype="float32"),
    ])
