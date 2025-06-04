"""Building localization step."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from ..models.localization import LocalizationModel
from ..data.datasets import LocalizationDataset
from ..utils.metrics import calculate_f1_score
from ..utils.visualization import save_debug_visualization

logger = logging.getLogger(__name__)

def run_localization(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    batch_size: int = 16,
    threshold: float = 0.5,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Run building localization on input images.
    
    Args:
        input_path: Path to input images
        output_path: Path to save results
        model_path: Path to model checkpoint
        batch_size: Batch size for inference
        threshold: Confidence threshold for predictions
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
        model = LocalizationModel.load_from_checkpoint(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise
    
    # Create dataset and dataloader
    try:
        dataset = LocalizationDataset(
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
                
                # Run inference with mixed precision
                with torch.cuda.amp.autocast():
                    predictions = model(images)
                    predictions = F.sigmoid(predictions)
                
                # Process predictions
                for idx, pred in enumerate(predictions):
                    # Convert to numpy
                    pred_np = pred.cpu().numpy()
                    
                    # Apply threshold
                    mask = (pred_np > threshold).astype(np.uint8)
                    
                    # Save debug visualization
                    if config.get('save_debug', False):
                        save_debug_visualization(
                            debug_path / f"batch_{batch_idx}_sample_{idx}.png",
                            batch['image'][idx].cpu().numpy(),
                            mask,
                            pred_np
                        )
                    
                    # Add to results
                    results.append({
                        'image': batch['image_path'][idx],
                        'mask': mask,
                        'confidence': float(pred_np.max())
                    })
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Save results
    try:
        np.save(output_path / 'localization_results.npy', results)
        logger.info(f"Saved results to {output_path / 'localization_results.npy'}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise 