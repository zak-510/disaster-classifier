"""Damage classification step."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm
import os
import rasterio
from rasterio.transform import rowcol
import cv2

from ..models.damage_classification import DamageClassifier
from ..data.datasets import DamageDataset
from ..utils.metrics import calculate_classification_metrics
from ..utils.visualization import save_damage_visualization

logger = logging.getLogger(__name__)

def extract_coordinates(image_path: Path, mask_path: Path) -> Tuple[float, float]:
    """Extract coordinates from image using Rasterio transform.
    
    Args:
        image_path: Path to the image file
        mask_path: Path to the mask file
        
    Returns:
        Tuple of (longitude, latitude)
    """
    try:
        with rasterio.open(image_path) as src:
            # Get center pixel coordinates
            height, width = src.height, src.width
            center_row, center_col = height // 2, width // 2
            
            # Convert to geographic coordinates
            lon, lat = src.xy(center_row, center_col)
            return float(lon), float(lat)
    except Exception as e:
        logger.warning(f"Failed to extract coordinates from {image_path}: {e}")
        return 0.0, 0.0

def run_classification(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    batch_size: int = 16,
    min_confidence: float = 0.7,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Run damage classification on localized buildings.
    
    Args:
        input_path: Path to input images and masks
        output_path: Path to save results
        model_path: Path to model checkpoint
        batch_size: Batch size for inference
        min_confidence: Minimum confidence threshold
        config: Optional configuration dictionary
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    debug_path = output_path / 'debug'
    debug_path.mkdir(exist_ok=True)
    
    # Load model
    try:
        model = DamageClassifier.load_from_checkpoint(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise
    
    # Create dataset and dataloader
    try:
        dataset = DamageDataset(
            input_path,
            transform=model.get_transform(),
            config=config
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True
        )
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise
    
    # Run inference
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing images")):
            try:
                # Move batch to device
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Run inference with mixed precision
                with torch.cuda.amp.autocast():
                    predictions = model(images, masks)
                    probabilities = F.softmax(predictions, dim=1)
                
                # Process predictions
                for idx, (prob, pred) in enumerate(zip(probabilities, predictions)):
                    # Get predicted class and confidence
                    pred_class = pred.argmax().item()
                    confidence = prob[pred_class].item()
                    
                    # Skip if confidence is too low
                    if confidence < min_confidence:
                        continue
                    
                    # Extract coordinates
                    lon, lat = extract_coordinates(
                        batch['image_path'][idx],
                        batch['mask_path'][idx]
                    )
                    
                    # Save debug visualization
                    if config.get('save_debug', False):
                        save_damage_visualization(
                            debug_path / f"batch_{batch_idx}_sample_{idx}.png",
                            batch['image'][idx].cpu().numpy(),
                            batch['mask'][idx].cpu().numpy(),
                            pred_class,
                            confidence
                        )
                    
                    # Add to results
                    results.append({
                        'image': str(batch['image_path'][idx]),
                        'mask': str(batch['mask_path'][idx]),
                        'damage_level': int(pred_class),
                        'confidence': float(confidence),
                        'lon': lon,
                        'lat': lat
                    })
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Save results
    try:
        # Save as JSON for easy parsing
        with open(output_path / 'classification_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as numpy array for compatibility
        np.save(output_path / 'classification_results.npy', results)
        
        logger.info(f"Saved results to {output_path / 'classification_results.json'}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise

def classify_damage(
    image_path: Path,
    mask: np.ndarray,
    model: DamageClassifier,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[int, float]:
    """
    Classify damage level for a single image and mask.
    
    Args:
        image_path: Path to the image
        mask: Binary mask of building locations
        model: Damage classification model
        device: Device to run inference on
        threshold: Confidence threshold
    
    Returns:
        Tuple of (damage_level, confidence)
    """
    # Load and preprocess image
    with rasterio.open(image_path) as src:
        image = src.read().transpose(1, 2, 0)  # (H, W, C)
    
    # Convert to tensor
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    image = image.to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(image)
        probabilities = F.softmax(predictions, dim=1)
        
        # Get predicted class and confidence
        pred_class = predictions.argmax(dim=1).item()
        confidence = probabilities[0, pred_class].item()
    
    return pred_class, confidence 