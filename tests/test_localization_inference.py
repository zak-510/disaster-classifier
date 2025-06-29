import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import sys

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.model import create_model
from shapely.wkt import loads

def load_localization_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model().to(device)
    
    checkpoint_path = os.path.join(project_root, 'weights/best_localization.pth')
    if not os.path.exists(checkpoint_path):
        print(f'ERROR: Localization model not found: {checkpoint_path}')
        return None, device
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f'SUCCESS: Localization model loaded from {checkpoint_path}')
    return model, device

def load_test_image_and_labels(image_id):
    base_path = os.path.join(project_root, f'Data/test/images/{image_id}_post_disaster.png')
    labels_path = os.path.join(project_root, f'Data/test/labels/{image_id}_post_disaster.json')
    
    if not os.path.exists(base_path) or not os.path.exists(labels_path):
        return None, None, None
    
    image = cv2.imread(base_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    ground_truth_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Handle the nested JSON structure: features -> xy -> list of features
    if 'features' in labels_data and 'xy' in labels_data['features']:
        features_list = labels_data['features']['xy']
        for feature in features_list:
            if feature['properties']['feature_type'] == 'building':
                wkt_str = feature.get('wkt')
                if wkt_str:
                    try:
                        geometry = loads(wkt_str)
                        coords = list(geometry.exterior.coords)
                        pts = np.array([[int(x), int(y)] for x, y in coords])
                        cv2.fillPoly(ground_truth_mask, [pts], 255)
                    except:
                        continue
    
    return image, ground_truth_mask, labels_data

def predict_localization(model, device, image):
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output).cpu().numpy()[0, 0]
    
    prediction_mask = (prediction > 0.5).astype(np.uint8) * 255
    return prediction_mask

def create_three_panel_visualization(image, ground_truth_mask, prediction_mask, image_id, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Original satellite image
    axes[0].imshow(image)
    axes[0].set_title('Original Satellite Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Ground truth masks - BLACK background with WHITE pixels for buildings
    axes[1].imshow(ground_truth_mask, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Ground Truth Building Masks', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Panel 3: Predicted masks - BLACK background with WHITE pixels for predictions  
    axes[2].imshow(prediction_mask, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Predicted Building Masks', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f'SUCCESS: Saved {output_path}')

def run_localization_test():
    print('LOCALIZATION MODEL INFERENCE TEST')
    print('=' * 60)
    
    model, device = load_localization_model()
    if model is None:
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
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(project_root, 'test_results/localization')
    os.makedirs(output_dir, exist_ok=True)
    
    for i, image_id in enumerate(test_images, 1):
        print(f'\nProcessing image {i}/{len(test_images)}: {image_id}')
        
        image, ground_truth_mask, labels_data = load_test_image_and_labels(image_id)
        if image is None:
            print(f'ERROR: Could not load {image_id}')
            continue
        
        prediction_mask = predict_localization(model, device, image)
        
        output_path = os.path.join(output_dir, f'localization_test_{i}.png')
        create_three_panel_visualization(image, ground_truth_mask, prediction_mask, image_id, output_path)
    
    print(f'\n' + '=' * 60)
    print('LOCALIZATION INFERENCE TEST COMPLETE')
    print(f'Visualizations saved to {output_dir}/')

if __name__ == '__main__':
    run_localization_test() 