import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def calculate_localization_metrics(pred_masks: torch.Tensor, true_masks: torch.Tensor,
                                threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate metrics for building localization.
    
    Args:
        pred_masks: Predicted masks (B, 1, H, W)
        true_masks: Ground truth masks (B, 1, H, W)
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary of metrics
    """
    pred_masks = (pred_masks > threshold).float()
    true_masks = true_masks.float()
    
    # Calculate IoU
    intersection = (pred_masks * true_masks).sum()
    union = pred_masks.sum() + true_masks.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Calculate F1 score
    pred_flat = pred_masks.view(-1).cpu().numpy()
    true_flat = true_masks.view(-1).cpu().numpy()
    f1 = f1_score(true_flat, pred_flat, average='binary')
    
    return {
        'iou': iou.item(),
        'f1_score': f1
    }

def calculate_classification_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate metrics for damage classification.
    
    Args:
        preds: Predicted class probabilities (B, C)
        targets: Ground truth class indices (B)
    
    Returns:
        Dictionary of metrics
    """
    pred_classes = preds.argmax(dim=1).cpu().numpy()
    target_classes = targets.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(target_classes, pred_classes)
    f1 = f1_score(target_classes, pred_classes, average='weighted')
    precision = precision_score(target_classes, pred_classes, average='weighted')
    recall = recall_score(target_classes, pred_classes, average='weighted')
    
    # Calculate per-class F1
    per_class_f1 = f1_score(target_classes, pred_classes, average=None)
    class_metrics = {
        f'f1_class_{i}': score for i, score in enumerate(per_class_f1)
    }
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        **class_metrics
    }

def calculate_damage_level_metrics(preds: torch.Tensor, targets: torch.Tensor,
                                 damage_levels: List[str]) -> Dict[str, float]:
    """
    Calculate metrics for damage level classification with class names.
    
    Args:
        preds: Predicted class probabilities (B, C)
        targets: Ground truth class indices (B)
        damage_levels: List of damage level names
    
    Returns:
        Dictionary of metrics with class names
    """
    metrics = calculate_classification_metrics(preds, targets)
    
    # Add class names to per-class metrics
    class_metrics = {
        f'f1_{damage_levels[i]}': metrics[f'f1_class_{i}']
        for i in range(len(damage_levels))
    }
    
    # Remove numeric class metrics
    metrics = {k: v for k, v in metrics.items() if not k.startswith('f1_class_')}
    
    return {**metrics, **class_metrics}

def calculate_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor,
                             num_classes: int) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        preds: Predicted class probabilities (B, C)
        targets: Ground truth class indices (B)
        num_classes: Number of classes
    
    Returns:
        Confusion matrix (C, C)
    """
    pred_classes = preds.argmax(dim=1).cpu().numpy()
    target_classes = targets.cpu().numpy()
    
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(target_classes, pred_classes):
        matrix[t, p] += 1
    
    return matrix 