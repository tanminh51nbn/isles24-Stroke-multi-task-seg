from monai.losses import TverskyLoss

def build_lesion_loss(alpha: float = 0.4, beta: float = 0.6):
    """
    Tversky Loss for Lesion Segmentation.
    Favors Recall (penalizes FN) with alpha=0.4, beta=0.6.
    """
    return TverskyLoss(
        include_background=True, 
        sigmoid=True, 
        alpha=alpha, 
        beta=beta,
        reduction='mean'
    )
