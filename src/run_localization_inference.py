import os
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import rasterio
from torch.utils.data import DataLoader
import cv2

from models.localization.unet import UNet
from data.xbd_dataset import XBDDataset
from utils.augmentations import get_localization_augmentations

def parse_args():
    parser = argparse.ArgumentParser(description='Run building localization inference')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--min_area', type=float, default=100, help='Minimum polygon area')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directories
    output_dir = Path(args.output_dir)
    masks_dir = output_dir / 'masks'
    geojson_dir = output_dir / 'geojson'
    masks_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = UNet(
        in_channels=3,
        out_channels=1,
        features=config['localization']['model'].get('features', [64, 128, 256, 512, 1024])
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataset and dataloader
    transform = get_localization_augmentations(config['localization']['data'], 'val')
    dataset = XBDDataset(args.input_dir, transform=transform, phase='val')
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Running inference'):
            images = batch['image'].to(device)
            image_paths = batch['image_path']
            
            # Forward pass
            preds = model.predict(images, threshold=args.threshold)
            
            # Process each prediction
            for i, (pred, image_path) in enumerate(zip(preds, image_paths)):
                # Convert to numpy
                pred = pred.cpu().numpy()[0]  # Remove batch and channel dimensions
                
                # Save mask
                mask_path = masks_dir / f"{Path(image_path).stem}_mask.png"
                cv2.imwrite(str(mask_path), (pred * 255).astype(np.uint8))
                
                # Convert to polygons and save GeoJSON
                polygons = XBDDataset.mask_to_polygons(pred, min_area=args.min_area)
                geojson_path = geojson_dir / f"{Path(image_path).stem}_buildings.json"
                with open(geojson_path, 'w') as f:
                    json.dump(polygons, f, indent=2)

if __name__ == '__main__':
    main() 