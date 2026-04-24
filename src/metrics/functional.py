import numpy as np
from scipy.ndimage import label, center_of_mass

def dice_3d(pred: np.ndarray, target: np.ndarray) -> float:
    """Computes 3D Dice score for binary arrays."""
    inter = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    if union == 0:
        return 1.0
    return (2.0 * inter) / union

def recall_3d(pred: np.ndarray, target: np.ndarray) -> float:
    """Computes 3D Recall for binary arrays."""
    tp = np.sum(pred * target)
    p = np.sum(target)
    if p == 0:
        return 1.0
    return tp / p

def object_f1_centroid(pred: np.ndarray, target: np.ndarray, radius: int = 3) -> float:
    """
    Object-level F1 score based on centroid matching (for LVO).
    If a predicted component's centroid is within `radius` of any GT component's centroid, it's a TP.
    """
    pred_labeled, num_pred = label(pred)
    target_labeled, num_target = label(target)
    
    if num_target == 0 and num_pred == 0:
        return 1.0
    if num_target == 0 or num_pred == 0:
        return 0.0
        
    pred_centers = center_of_mass(pred, pred_labeled, range(1, num_pred + 1))
    target_centers = center_of_mass(target, target_labeled, range(1, num_target + 1))
    
    # Convert to valid centers
    pred_centers = [c for c in pred_centers if not np.isnan(c[0])]
    target_centers = [c for c in target_centers if not np.isnan(c[0])]
    
    tp = 0
    matched_targets = set()
    
    for p_c in pred_centers:
        for i, t_c in enumerate(target_centers):
            if i in matched_targets:
                continue
            dist = np.linalg.norm(np.array(p_c) - np.array(t_c))
            if dist <= radius:
                tp += 1
                matched_targets.add(i)
                break
                
    fp = len(pred_centers) - tp
    fn = len(target_centers) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)
