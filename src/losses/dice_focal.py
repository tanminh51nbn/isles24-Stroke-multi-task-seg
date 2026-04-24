from monai.losses import TverskyLoss

def build_cow_loss(alpha: float = 0.5, beta: float = 0.5):
    """
    Tversky Loss (equivalent to Dice with 0.5/0.5) for CoW Segmentation.
    Switched from DiceFocal to maintain scale consistency with other tasks.
    """
    return TverskyLoss(
        include_background=True, 
        sigmoid=True, 
        alpha=alpha, 
        beta=beta,
        reduction='mean'
    )
