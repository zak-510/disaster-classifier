import os
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
import json
import csv
from tqdm import tqdm
import rasterio
from torch.utils.data import DataLoader
import cv2
from shapely.geometry import box, mapping
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

from models.localization.unet import UNet
from data.xbd_dataset import XBDDataset
from data.building_pair_dataset import BuildingPairDataset
from utils.augmentations import get_localization_augmentations
from models.damage_classification import DamageClassifier
from utils.report_generator import HTMLReportGenerator

def parse_args():
    parser = argparse.ArgumentParser(description='Run full xBD pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--localization_checkpoint', type=str, required=True, help='Path to localization model checkpoint')
    parser.add_argument('--classification_checkpoint', type=str, required=True, help='Path to damage classification model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input images directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Localization confidence threshold')
    parser.add_argument('--min_area', type=float, default=100, help='Minimum building area')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for classification')
    return parser.parse_args()

def load_image(image_path):
    """Load image using rasterio."""
    with rasterio.open(image_path) as src:
        return src.read().transpose(1, 2, 0)  # (H, W, C)

def crop_building(image, bbox, padding=10):
    """Crop building from image with padding."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image[y1:y2, x1:x2]

def create_visualization(image, buildings, damage_levels, confidences, output_path):
    """Create visualization with color-coded damage levels and confidence scores."""
    # Color mapping for damage levels
    colors = {
        0: (0, 255, 0),    # Green - No damage
        1: (255, 255, 0),  # Yellow - Minor damage
        2: (255, 165, 0),  # Orange - Major damage
        3: (255, 0, 0)     # Red - Destroyed
    }
    
    # Create overlay
    overlay = image.copy()
    for bbox, damage, conf in zip(buildings, damage_levels, confidences):
        x1, y1, x2, y2 = bbox
        color = colors[damage]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Add damage level and confidence text
        text = f"{damage} ({conf:.2f})"
        cv2.putText(overlay, text, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Blend with original image
    alpha = 0.7
    output = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    cv2.imwrite(str(output_path), output)

def process_buildings_batch(classification_model, dataloader, device, config):
    """Process a batch of building pairs and return predictions with confidence scores."""
    all_predictions = []
    all_confidences = []
    all_indices = []
    
    # Get confidence thresholds from config
    min_confidence = config['classification']['inference']['min_confidence']
    damage_thresholds = {
        i: level['min_confidence']
        for i, level in enumerate(config['classification']['inference']['damage_levels'])
    }
    
    with torch.no_grad():
        for batch in dataloader:
            image_pairs = batch['image_pair'].to(device)
            indices = batch['idx']
            
            # Forward pass
            logits = classification_model(image_pairs)
            probs = F.softmax(logits, dim=1)
            
            # Get predictions and confidence scores
            damage_levels = torch.argmax(probs, dim=1)
            confidences = torch.max(probs, dim=1)[0]
            
            # Apply confidence thresholds
            valid_mask = torch.ones_like(confidences, dtype=torch.bool)
            for i, threshold in damage_thresholds.items():
                class_mask = damage_levels == i
                valid_mask[class_mask] = confidences[class_mask] >= threshold
            
            # Store valid predictions
            valid_indices = indices[valid_mask]
            valid_predictions = damage_levels[valid_mask]
            valid_confidences = confidences[valid_mask]
            
            all_predictions.extend(valid_predictions.cpu().numpy())
            all_confidences.extend(valid_confidences.cpu().numpy())
            all_indices.extend(valid_indices.cpu().numpy())
    
    # Sort predictions by original index
    sorted_pairs = sorted(zip(all_indices, all_predictions, all_confidences))
    return [(pred, conf) for _, pred, conf in sorted_pairs]

def generate_summary_report(results, config, output_dir, disaster_name):
    """Generate a comprehensive summary report of filtered predictions."""
    # Initialize statistics
    stats = {
        'total_detected': len(results),
        'damage_levels': defaultdict(lambda: {'count': 0, 'confidences': []}),
        'filtered': defaultdict(int),
        'thresholds': {
            i: level['min_confidence']
            for i, level in enumerate(config['classification']['inference']['damage_levels'])
        }
    }
    
    # Collect statistics
    for result in results:
        damage_level = result['damage_level']
        confidence = result['confidence']
        
        stats['damage_levels'][damage_level]['count'] += 1
        stats['damage_levels'][damage_level]['confidences'].append(confidence)
    
    # Calculate additional statistics
    for level in stats['damage_levels']:
        confidences = stats['damage_levels'][level]['confidences']
        stats['damage_levels'][level].update({
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'std_confidence': np.std(confidences)
        })
    
    # Generate HTML report
    report_generator = HTMLReportGenerator(output_dir, disaster_name)
    
    # Collect example images
    example_images = []
    viz_dir = output_dir / 'visualizations'
    if viz_dir.exists():
        for img_path in viz_dir.glob('*_damage.png'):
            example_images.append((img_path, f"Damage Assessment: {img_path.stem}"))
    
    # Generate report
    report_path = report_generator.generate_report(stats, example_images)
    
    # Print summary to console
    print(f"\nSummary for {disaster_name}:")
    print(f"Total buildings detected: {stats['total_detected']}")
    print(f"Buildings retained: {sum(d['count'] for d in stats['damage_levels'].values())}")
    print(f"Buildings filtered out: {stats['total_detected'] - sum(d['count'] for d in stats['damage_levels'].values())}")
    print("\nPer-class statistics:")
    for level, name in enumerate(['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']):
        if level in stats['damage_levels']:
            data = stats['damage_levels'][level]
            print(f"\n{name}:")
            print(f"  Count: {data['count']}")
            print(f"  Average confidence: {data['avg_confidence']:.4f}")
            print(f"  Min confidence: {data['min_confidence']:.4f}")
            print(f"  Max confidence: {data['max_confidence']:.4f}")
    
    print(f"\nInteractive HTML report generated: {report_path}")
    
    return stats

def main():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directories
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / 'crops'
    viz_dir = output_dir / 'visualizations'
    crops_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    localization_model = UNet(
        in_channels=3,
        out_channels=1,
        features=config['localization']['model'].get('features', [64, 128, 256, 512, 1024])
    ).to(device)
    
    classification_model = DamageClassifier(
        num_classes=4,
        model_name=config['classification']['model'].get('name', 'resnet50')
    ).to(device)
    
    # Load checkpoints
    localization_model.load_state_dict(torch.load(args.localization_checkpoint, map_location=device)['model_state_dict'])
    classification_model.load_state_dict(torch.load(args.classification_checkpoint, map_location=device)['model_state_dict'])
    
    localization_model.eval()
    classification_model.eval()
    
    # Process each disaster directory
    for disaster_dir in Path(args.input_dir).iterdir():
        if not disaster_dir.is_dir():
            continue
            
        print(f"Processing {disaster_dir.name}...")
        
        # Get pre and post disaster images
        pre_images = list(disaster_dir.glob('images/*_pre_disaster.png'))
        post_images = list(disaster_dir.glob('images/*_post_disaster.png'))
        
        # Match pre and post images
        image_pairs = []
        for pre_img in pre_images:
            post_img = pre_img.parent / f"{pre_img.stem.replace('_pre_disaster', '_post_disaster')}.png"
            if post_img.exists():
                image_pairs.append((pre_img, post_img))
        
        # Process each image pair
        results = []
        for pre_img, post_img in tqdm(image_pairs, desc="Processing images"):
            # Load images
            pre_image = load_image(pre_img)
            post_image = load_image(post_img)
            
            # Run localization on pre-disaster image
            transform = get_localization_augmentations(config['localization']['data'], 'val')
            transformed = transform(image=pre_image)
            pre_tensor = torch.from_numpy(transformed['image']).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_mask = localization_model.predict(pre_tensor, threshold=args.threshold)
                pred_mask = pred_mask[0, 0].cpu().numpy()
            
            # Get building polygons
            buildings = XBDDataset.mask_to_polygons(pred_mask, min_area=args.min_area)
            
            if not buildings['features']:
                continue
            
            # Prepare data for batch processing
            pre_images_list = [pre_image] * len(buildings['features'])
            post_images_list = [post_image] * len(buildings['features'])
            bboxes = []
            geometries = []
            
            for feature in buildings['features']:
                bbox = box(*feature['geometry']['coordinates'][0]).bounds
                bbox = tuple(map(int, bbox))  # (x1, y1, x2, y2)
                bboxes.append(bbox)
                geometries.append(feature['geometry'])
            
            # Create dataset and dataloader for batch processing
            dataset = BuildingPairDataset(
                pre_images_list,
                post_images_list,
                bboxes,
                transform=transform
            )
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Process buildings in batches
            predictions = process_buildings_batch(classification_model, dataloader, device, config)
            
            # Combine results
            building_results = [
                {
                    'bbox': bbox,
                    'damage_level': pred,
                    'confidence': conf,
                    'geometry': geometry
                }
                for (bbox, geometry), (pred, conf) in zip(zip(bboxes, geometries), predictions)
            ]
            
            # Create visualization
            create_visualization(
                post_image,
                [b['bbox'] for b in building_results],
                [b['damage_level'] for b in building_results],
                [b['confidence'] for b in building_results],
                viz_dir / f"{post_img.stem}_damage.png"
            )
            
            # Add to results
            results.extend(building_results)
        
        # Generate summary report
        stats = generate_summary_report(results, config, output_dir, disaster_dir.name)
        
        # Save results
        # 1. CSV
        csv_path = output_dir / f"{disaster_dir.name}_damage_assessment.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'x1', 'y1', 'x2', 'y2', 'damage_level', 'confidence'])
            for result in results:
                writer.writerow([
                    post_img.name,
                    *result['bbox'],
                    result['damage_level'],
                    f"{result['confidence']:.4f}"
                ])
        
        # 2. GeoJSON
        geojson_path = output_dir / f"{disaster_dir.name}_damage_assessment.json"
        geojson = {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'geometry': result['geometry'],
                    'properties': {
                        'damage_level': result['damage_level'],
                        'confidence': float(f"{result['confidence']:.4f}")
                    }
                }
                for result in results
            ]
        }
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f, indent=2)

if __name__ == '__main__':
    main() 