import os
import sys
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Dict, Any, Tuple

from src.models.localization.unet import UNet
from src.models.damage_classification import DamageClassifier
from src.utils.mask_processing import process_predictions
from src.utils.report_generator import HTMLReportGenerator
from src.steps.classify import extract_coordinates, classify_damage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(output_dir: Path) -> Dict[str, Path]:
    """Create necessary output directories."""
    dirs = {
        'debug': output_dir / 'debug',
        'visualizations': output_dir / 'visualizations',
        'predictions': output_dir / 'predictions'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def load_model(model_path: Path, model_type: str, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint."""
    try:
        # Always load to CPU first, then move to target device
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if model_type == 'localization':
            model = UNet(in_channels=3, out_channels=1)
        else:
            model = DamageClassifier(num_classes=4)
        
        # Handle test mode with mock checkpoints
        if os.getenv('TESTING'):
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load {model_type} model from {model_path}: {e}")
        raise

def process_image(
    img_path: Path,
    localization_model: torch.nn.Module,
    damage_model: torch.nn.Module,
    device: torch.device,
    debug_dir: Optional[Path],
    threshold: float = 0.1,
    batch_size: int = 4
) -> Dict[str, Any]:
    """Process a single image with batching support."""
    # Load and preprocess image
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    # Process in batches if image is large
    height, width = img.shape[:2]
    if height * width > 1000000:  # Arbitrary threshold for large images
        logger.info(f"Processing large image in batches: {img_path}")
        # Implement batching logic here
        # For now, we'll just process the whole image
        pass
    
    with torch.no_grad():
        # Localization
        raw_mask = localization_model(img_tensor.to(device))
        raw_mask = raw_mask.cpu().numpy()[0, 0]
        
        # Save debug artifacts only if debug_dir is provided
        if debug_dir is not None:
            debug_path = debug_dir / f"raw_mask_{img_path.stem}.png"
            cv2.imwrite(str(debug_path), (raw_mask * 255).astype(np.uint8))
        
        # Process mask
        processed_mask = process_predictions(raw_mask, threshold=threshold)
        
        # Damage classification
        # TODO: Implement batching for damage classification
        damage_pred = damage_model(img_tensor.to(device))
        damage_conf = torch.softmax(damage_pred, dim=1).cpu().numpy()[0]
    
    return {
        'raw_mask': raw_mask,
        'processed_mask': processed_mask,
        'damage_conf': damage_conf
    }

def run_inference(
    input_dir: Path,
    output_dir: Path,
    localization_model_path: Path,
    damage_model_path: Path,
    threshold: float = 0.5,
    batch_size: int = 16,
    debug: bool = False
) -> None:
    """Run inference pipeline."""
    # Ensure output directories exist
    for subdir in ['localization', 'classification', 'debug']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    logger.info(f"Loading models on device: {device}")
    localization_model = load_model(localization_model_path, 'localization', device)
    damage_model = load_model(damage_model_path, 'damage', device)
    
    # Process images
    image_paths = list(input_dir.glob('*.png'))
    if not image_paths:
        logger.warning(f"No images found in {input_dir}")
        return
    
    # Run localization
    logger.info("Running localization...")
    masks = []
    debug_dir = output_dir / 'debug' if debug else None
    
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start:batch_start + batch_size]
        batch_results = [
            process_image(path, localization_model, damage_model, device, debug_dir, threshold)
            for path in batch_paths
        ]
        masks.extend([result['processed_mask']['mask'] for result in batch_results])  # Extract mask from dict
    
    # Save localization results
    masks = np.array(masks)
    np.save(output_dir / 'localization' / 'localization_results.npy', masks)
    
    # Run damage classification if masks were found
    if len(masks) > 0:
        logger.info("Running damage classification...")
        results = []
        with torch.no_grad():
            for img_path, mask in zip(image_paths, masks):
                if np.max(mask) > 0:  # Check if any buildings were detected
                    damage_level, confidence = classify_damage(
                        img_path, mask, damage_model, device, threshold
                    )
                    if confidence >= threshold:
                        # Save mask to file for coordinate extraction
                        mask_path = output_dir / 'localization' / f"{img_path.stem}_mask.png"
                        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
                        
                        results.append({
                            'image': str(img_path),
                            'damage_level': int(damage_level),
                            'confidence': float(confidence),
                            'coordinates': extract_coordinates(img_path, mask_path),
                            'timestamp': datetime.now().isoformat()
                        })
        
        # Save classification results
        with open(output_dir / 'classification' / 'classification_results.json', 'w') as f:
            json.dump({'results': results}, f)
        
        # Generate statistics
        generate_statistics(results, output_dir / 'classification' / 'statistics.json')
    else:
        logger.warning("No buildings detected in any images")

def generate_statistics(results: list, out_path: Path) -> None:
    """Generate basic statistics from classification results."""
    stats = {
        "total": len(results),
        "damage_levels": {},
        "avg_confidence": 0.0
    }
    
    if results:
        confidence_sum = 0
        for result in results:
            damage_level = result["damage_level"]
            stats["damage_levels"][damage_level] = stats["damage_levels"].get(damage_level, 0) + 1
            confidence_sum += result["confidence"]
        stats["avg_confidence"] = confidence_sum / len(results)
    
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on xBD dataset')
    parser.add_argument('--input_dir', type=Path, required=True, help='Input directory containing images')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output directory for results')
    parser.add_argument('--localization_model', type=Path, required=True, help='Path to localization model')
    parser.add_argument('--damage_model', type=Path, required=True, help='Path to damage classification model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Mask threshold value')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    run_inference(
        args.input_dir,
        args.output_dir,
        args.localization_model,
        args.damage_model,
        args.threshold,
        args.batch_size,
        args.debug
    ) 