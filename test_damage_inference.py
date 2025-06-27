import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from model import create_model
from damage_model import create_damage_model
from shapely.wkt import loads
from skimage import measure

def load_models():
    device = torch.device('cuda')
    
    # Load localization model
    loc_model = create_model().to(device)
    loc_checkpoint_path = './checkpoints/extended/model_epoch_20.pth'
    
    if not os.path.exists(loc_checkpoint_path):
        print(f'ERROR: Localization model not found: {loc_checkpoint_path}')
        return None, None, device
    
    loc_checkpoint = torch.load(loc_checkpoint_path)
    if isinstance(loc_checkpoint, dict) and 'model_state_dict' in loc_checkpoint:
        loc_model.load_state_dict(loc_checkpoint['model_state_dict'])
    else:
        loc_model.load_state_dict(loc_checkpoint)
    loc_model.eval()
    
    # Load damage model
    damage_model = create_damage_model().to(device)
    damage_checkpoint_path = './weights/best_damage_model_optimized.pth'
    
    if not os.path.exists(damage_checkpoint_path):
        print(f'ERROR: Damage model not found: {damage_checkpoint_path}')
        return None, None, device
    
    damage_checkpoint = torch.load(damage_checkpoint_path)
    damage_model.load_state_dict(damage_checkpoint['model_state_dict'])
    damage_model.eval()
    
    print(f'SUCCESS: Models loaded successfully')
    return loc_model, damage_model, device

def load_test_image_and_labels(image_id):
    base_path = f'./Data/test/images/{image_id}_post_disaster.png'
    labels_path = f'./Data/test/labels/{image_id}_post_disaster.json'
    
    if not os.path.exists(base_path) or not os.path.exists(labels_path):
        return None, None
    
    image = cv2.imread(base_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    return image, labels_data

def detect_buildings(loc_model, device, image):
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    
    with torch.no_grad():
        output = loc_model(image_tensor)
        prediction = torch.sigmoid(output).cpu().numpy()[0, 0]
    
    # Apply threshold and find connected components
    binary_prediction = (prediction > 0.5).astype(np.uint8)
    labeled_regions = measure.label(binary_prediction)
    regions = measure.regionprops(labeled_regions)
    
    building_regions = []
    for region in regions:
        if region.area > 100:  # Minimum area threshold
            minr, minc, maxr, maxc = region.bbox
            building_regions.append({
                'bbox': (minr, minc, maxr, maxc),
                'mask': labeled_regions == region.label
            })
    
    return building_regions

def get_ground_truth_buildings(labels_data, image_shape):
    """Extract ground truth building regions from JSON labels"""
    ground_truth_buildings = []
    
    if 'features' in labels_data and 'xy' in labels_data['features']:
        features_list = labels_data['features']['xy']
        for feature in features_list:
            if feature['properties']['feature_type'] == 'building':
                wkt_str = feature.get('wkt')
                damage_type = feature.get('properties', {}).get('subtype', 'no-damage')
                
                if wkt_str:
                    try:
                        geometry = loads(wkt_str)
                        coords = list(geometry.exterior.coords)
                        pts = np.array([[int(x), int(y)] for x, y in coords])
                        
                        # Create mask for this building
                        mask = np.zeros(image_shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask, [pts], 1)
                        mask = mask.astype(bool)
                        
                        # Get bounding box from polygon coordinates
                        x_coords = pts[:, 0]
                        y_coords = pts[:, 1]
                        minc, maxc = int(np.min(x_coords)), int(np.max(x_coords))
                        minr, maxr = int(np.min(y_coords)), int(np.max(y_coords))
                        
                        # Map damage type to number
                        damage_types = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3}
                        damage_value = damage_types.get(damage_type, 0)
                        
                        ground_truth_buildings.append({
                            'bbox': (minr, minc, maxr, maxc),
                            'mask': mask,
                            'damage_type': damage_value
                        })
                    except:
                        continue
    
    return ground_truth_buildings

def predict_damage(model, device, patch):
    patch_tensor = torch.from_numpy(patch.astype(np.float32) / 255.0)
    patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(patch_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted].item()
    
    return predicted.item(), confidence

def get_ground_truth_damage(labels_data, bbox):
    damage_types = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3}
    minr, minc, maxr, maxc = bbox
    
    # Handle the nested JSON structure: features -> xy -> list of features
    if 'features' in labels_data and 'xy' in labels_data['features']:
        features_list = labels_data['features']['xy']
        for feature in features_list:
            if feature['properties']['feature_type'] == 'building':
                wkt_str = feature.get('wkt')
                damage_type = feature.get('properties', {}).get('subtype', 'no-damage')
                
                if wkt_str:
                    try:
                        geometry = loads(wkt_str)
                        if hasattr(geometry, 'bounds'):
                            gminx, gminy, gmaxx, gmaxy = geometry.bounds
                            # Check if building region overlaps with ground truth
                            if not (maxc < gminx or minc > gmaxx or maxr < gminy or minr > gmaxy):
                                return damage_types.get(damage_type, 0)
                    except:
                        continue
    
    return 0  # Default to no-damage

def create_damage_visualization(image, ground_truth_buildings, predicted_building_regions, predicted_damages, image_id, output_path):
    class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
    colors = [(0, 255, 0), (255, 255, 0), (255, 165, 0), (255, 0, 0)]  # Green, Yellow, Orange, Red
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Original satellite image
    axes[0].imshow(image)
    axes[0].set_title('Original Satellite Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Ground truth damage - SATELLITE IMAGE background with colored building pixels (using ground truth buildings)
    gt_image = image.copy()
    for building in ground_truth_buildings:
        mask = building['mask']
        gt_damage = building['damage_type']
        color = colors[gt_damage]
        
        # Color the building pixels based on ground truth damage class
        gt_image[mask] = color
    
    axes[1].imshow(gt_image)
    axes[1].set_title('Ground Truth Damage Classification', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Panel 3: Predicted damage - SATELLITE IMAGE background with colored building pixels (using predicted buildings)
    pred_image = image.copy()
    for i, (region, pred_damage) in enumerate(zip(predicted_building_regions, predicted_damages)):
        mask = region['mask']
        color = colors[pred_damage]
        
        # Color the building pixels based on predicted damage class
        pred_image[mask] = color
    
    axes[2].imshow(pred_image)
    axes[2].set_title('Predicted Damage Classification', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f'SUCCESS: Saved {output_path}')

def process_image(loc_model, damage_model, device, image, labels_data, image_id, output_path):
    # Get ground truth building regions
    ground_truth_buildings = get_ground_truth_buildings(labels_data, image.shape)
    
    # Detect buildings using localization model for predictions
    predicted_building_regions = detect_buildings(loc_model, device, image)
    
    if len(predicted_building_regions) == 0:
        print(f'WARNING: No buildings detected in {image_id}')
        return
    
    predicted_damages = []
    
    # Process each predicted building for damage classification
    for region in predicted_building_regions:
        bbox = region['bbox']
        minr, minc, maxr, maxc = bbox
        
        # Extract building patch
        patch = image[minr:maxr, minc:maxc]
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            # Resize to model input size
            patch_resized = cv2.resize(patch, (64, 64))
            
            # Predict damage
            pred_damage, confidence = predict_damage(damage_model, device, patch_resized)
            predicted_damages.append(pred_damage)
    
    # Create visualization with separate ground truth and predicted buildings
    create_damage_visualization(image, ground_truth_buildings, predicted_building_regions, 
                              predicted_damages, image_id, output_path)
    
    print(f'Processed {len(ground_truth_buildings)} ground truth buildings and {len(predicted_building_regions)} predicted buildings in {image_id}')

def run_damage_test():
    print('DAMAGE CLASSIFICATION INFERENCE TEST')
    print('=' * 60)
    
    loc_model, damage_model, device = load_models()
    if loc_model is None or damage_model is None:
        return
    
    # Select 10 test images spanning multiple disasters
    test_images = [
        'palu-tsunami_00000181',
        'hurricane-michael_00000366',
        'santa-rosa-wildfire_00000089',
        'hurricane-florence_00000013',
        'socal-fire_00001400',
        'socal-fire_00001372',
        'hurricane-michael_00000437',
        'hurricane-michael_00000399',
        'hurricane-florence_00000095',
        'hurricane-florence_00000087'
    ]
    
    print(f'Processing {len(test_images)} test images...')
    
    for i, image_id in enumerate(test_images, 1):
        print(f'\nProcessing image {i}/{len(test_images)}: {image_id}')
        
        image, labels_data = load_test_image_and_labels(image_id)
        if image is None:
            print(f'ERROR: Could not load {image_id}')
            continue
        
        output_path = f'test_results/damage/damage_test_{i}.png'
        process_image(loc_model, damage_model, device, image, labels_data, image_id, output_path)
    
    print(f'\n' + '=' * 60)
    print('DAMAGE CLASSIFICATION INFERENCE TEST COMPLETE')
    print('Generated files:')
    for idx in range(1, len(test_images) + 1):
        print(f'  - test_results/damage/damage_test_{idx}.png')

if __name__ == '__main__':
    run_damage_test() 