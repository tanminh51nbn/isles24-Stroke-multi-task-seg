from monai.losses import TverskyLoss

def build_tversky_loss(alpha: float = 0.5, beta: float = 0.5):
    """
    Generic Tversky Loss.
    """
    return TverskyLoss(
        include_background=True, 
        sigmoid=True, 
        alpha=alpha, 
        beta=beta,
        reduction='mean'
    )
