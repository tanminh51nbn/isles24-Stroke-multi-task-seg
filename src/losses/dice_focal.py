from monai.losses import DiceFocalLoss

def build_cow_loss(lambda_dice: float = 1.0, lambda_focal: float = 1.0, gamma: float = 2.0):
    """
    Dice Focal Loss for CoW Segmentation.
    Stable structures with focal component to handle difficulty.
    """
    return DiceFocalLoss(
        include_background=True, 
        sigmoid=True, 
        lambda_dice=lambda_dice, 
        lambda_focal=lambda_focal, 
        gamma=gamma,
        reduction='mean'
    )
