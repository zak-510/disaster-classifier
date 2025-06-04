import numpy as np
import torch
from typing import Dict, Any, Tuple, Union
import cv2

def process_predictions(
    pred_mask: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    min_size: int = 100
) -> Dict[str, Any]:
    """
    Process model predictions to extract building masks and metadata.
    
    Args:
        pred_mask: Raw prediction mask from model (torch.Tensor or numpy.ndarray)
        threshold: Confidence threshold for mask binarization
        min_size: Minimum size of building instances to keep
        
    Returns:
        Dictionary containing processed masks and metadata
    """
    # Convert to numpy if torch tensor
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.squeeze().cpu().numpy()
    
    # Binarize mask
    binary_mask = (pred_mask > threshold).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    # Filter small components
    for i in range(1, num_labels):
        if np.sum(labels == i) < min_size:
            labels[labels == i] = 0
    
    # Get component properties
    props = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Extract metadata
    metadata = {
        'num_buildings': num_labels - 1,  # Subtract background
        'avg_confidence': float(np.mean(pred_mask[binary_mask > 0])),
        'min_confidence': float(np.min(pred_mask[binary_mask > 0])),
        'max_confidence': float(np.max(pred_mask[binary_mask > 0])),
        'building_sizes': [int(prop[4]) for prop in props[2][1:]]  # Skip background
    }
    
    return {
        'mask': labels,
        'metadata': metadata
    } 