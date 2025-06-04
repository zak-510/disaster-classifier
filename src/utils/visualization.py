"""Visualization utilities for xBD pipeline."""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple

def save_damage_visualization(
    output_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    damage_level: int,
    confidence: float,
    alpha: float = 0.5,
    damage_colors: Optional[dict] = None
) -> None:
    """
    Save visualization of damage prediction.
    
    Args:
        output_path: Path to save visualization
        image: Input image (H, W, C)
        mask: Binary mask (H, W)
        damage_level: Predicted damage level
        confidence: Prediction confidence
        alpha: Transparency for overlay
        damage_colors: Optional dict mapping damage levels to colors
    """
    if damage_colors is None:
        damage_colors = {
            0: (0, 255, 0),    # No damage (green)
            1: (255, 255, 0),  # Minor damage (yellow)
            2: (255, 128, 0),  # Major damage (orange)
            3: (255, 0, 0)     # Destroyed (red)
        }
    
    # Ensure image is uint8 and in BGR
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create overlay
    overlay = image.copy()
    color = damage_colors.get(damage_level, (128, 128, 128))
    overlay[mask > 0] = color
    
    # Blend images
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Add text
    text = f"Damage: {damage_level} (conf: {confidence:.2f})"
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save
    cv2.imwrite(str(output_path), output) 