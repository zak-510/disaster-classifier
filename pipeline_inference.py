import torch
import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from shapely.wkt import loads
from shapely.geometry import Polygon as ShapelyPolygon
from damage_model import create_damage_model
from model import create_model
from skimage import measure

def load_models():
    device = torch.device('cuda')
    
    # Load localization model
    loc_model = create_model().to(device)
    loc_checkpoint_path = 'checkpoints/extended/model_epoch_20.pth'
    if os.path.exists(loc_checkpoint_path):
        loc_checkpoint = torch.load(loc_checkpoint_path)
        # Check if it's a structured checkpoint or raw weights
        if isinstance(loc_checkpoint, dict) and 'model_state_dict' in loc_checkpoint:
            loc_model.load_state_dict(loc_checkpoint['model_state_dict'])
        else:
            # Direct state dict loading
            loc_model.load_state_dict(loc_checkpoint)
        print(f'Loaded localization model from {loc_checkpoint_path}')
    else:
        print(f'Localization model checkpoint not found: {loc_checkpoint_path}')
        return None
    
    # Load damage classification model
    damage_model = create_damage_model().to(device)
    damage_checkpoint_path = 'weights/best_damage_model_optimized.pth'
    if os.path.exists(damage_checkpoint_path):
        damage_checkpoint = torch.load(damage_checkpoint_path)
        damage_model.load_state_dict(damage_checkpoint['model_state_dict'])
        print(f'Loaded damage model from {damage_checkpoint_path}')
        print(f'Best accuracy: {damage_checkpoint["best_acc"]:.4f}')
    else:
        print(f'Damage model checkpoint not found: {damage_checkpoint_path}')
        return None
    
    loc_model.eval()
    damage_model.eval()
    return loc_model, damage_model, device

def extract_patch(image, wkt_str, patch_size=64):
    try:
        geom = loads(wkt_str)
        if isinstance(geom, ShapelyPolygon) and geom.is_valid:
            coords = np.array(geom.exterior.coords, dtype=np.int32)
            
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            half_size = patch_size // 2
            
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(image.shape[1], center_x + half_size)
            y2 = min(image.shape[0], center_y + half_size)
            
            patch = image[y1:y2, x1:x2]
            
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                patch = cv2.resize(patch, (patch_size, patch_size))
                return patch, coords
        return None, None
    except:
        return None, None

def detect_buildings(loc_model, device, image):
    # Prepare image for localization model
    image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mask_output = loc_model(image_tensor)
        mask_output = torch.sigmoid(mask_output)
        binary_mask = (mask_output > 0.5).cpu().numpy()[0, 0]
    
    # Return the full binary mask and individual regions for damage classification
    labeled_mask = measure.label(binary_mask, connectivity=2)
    regions = measure.regionprops(labeled_mask)
    
    region_data = []
    for region in regions:
        if region.area > 50:  # Filter small detections
            # Get the actual mask pixels for this region
            region_mask = (labeled_mask == region.label)
            region_data.append({
                'mask': region_mask,
                'bbox': region.bbox,  # (min_row, min_col, max_row, max_col)
                'area': region.area,
                'centroid': region.centroid
            })
    
    return binary_mask, region_data



def predict_damage(model, device, patch):
    patch_tensor = torch.from_numpy(patch.astype(np.float32) / 255.0)
    patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(patch_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted].item()
    
    return predicted.item(), confidence

def get_damage_colors():
    return {
        'no-damage': (0, 255, 0),      # Green
        'minor-damage': (255, 255, 0), # Yellow
        'major-damage': (255, 165, 0), # Orange
        'destroyed': (255, 0, 0)       # Red
    }

def process_image(image_path, label_path, loc_model, damage_model, device, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != (1024, 1024):
        image = cv2.resize(image, (1024, 1024))
    
    try:
        with open(label_path, 'r') as f:
            data = json.load(f)
    except:
        return None
    
    if 'features' not in data or 'xy' not in data['features']:
        return None
    
    features = data['features']['xy']
    damage_classes = {
        'no-damage': 0,
        'minor-damage': 1,
        'major-damage': 2,
        'destroyed': 3
    }
    class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
    colors = get_damage_colors()
    
    # Create three copies of the image for visualization
    img_original = image.copy()
    img_ground_truth = image.copy()
    img_prediction = image.copy()
    
    # Process ground truth buildings for Panel 2
    gt_buildings = []
    for feature in features:
        if 'wkt' in feature and 'properties' in feature:
            props = feature['properties']
            if props.get('feature_type') == 'building':
                damage_type = props.get('subtype', 'no-damage')
                
                if damage_type in damage_classes:
                    patch, coords = extract_patch(image, feature['wkt'])
                    if patch is not None and coords is not None:
                        gt_buildings.append({
                            'coords': coords,
                            'damage_type': damage_type
                        })
                        
                        # Draw polygons on ground truth image
                        gt_color = colors[damage_type]
                        cv2.fillPoly(img_ground_truth, [coords], gt_color)
                        cv2.polylines(img_ground_truth, [coords], True, (0, 0, 0), 2)
    
    # Use localization model to detect buildings for Panel 3
    print(f"Detecting buildings using localization model...")
    binary_mask, region_data = detect_buildings(loc_model, device, image)
    print(f"Found {len(region_data)} buildings")
    
    # Start with satellite image background
    img_prediction = image.copy()
    
    # Classify each detected region and color the mask pixels
    detected_with_damage = []
    for region in region_data:
        # Extract patch from bounding box for classification
        bbox = region['bbox']  # (min_row, min_col, max_row, max_col)
        patch = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            # Resize to standard patch size
            patch_resized = cv2.resize(patch, (64, 64))
            pred_class, confidence = predict_damage(damage_model, device, patch_resized)
            pred_damage_type = class_names[pred_class]
            
            detected_with_damage.append({
                'mask': region['mask'],
                'predicted_damage': pred_damage_type,
                'confidence': confidence,
                'bbox': bbox
            })
            
            # Color the actual mask pixels with damage classification color
            pred_color = colors[pred_damage_type]
            mask_pixels = region['mask']
            img_prediction[mask_pixels] = pred_color
    
    # Calculate accuracy by comparing with ground truth
    correct_predictions = 0
    total_comparisons = 0
    
    for gt_building in gt_buildings:
        gt_coords = gt_building['coords']
        gt_damage = gt_building['damage_type']
        gt_center = gt_coords.mean(axis=0)
        
        # Find closest detected building by checking overlap with detected masks
        best_overlap = 0
        closest_detection = None
        
        for detected in detected_with_damage:
            bbox = detected['bbox']
            bbox_center = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
            distance = np.linalg.norm(gt_center - bbox_center[::-1])  # bbox is (row,col), gt_center is (x,y)
            
            if distance < 100:  # 100 pixel threshold for matching
                overlap = distance  # Use inverse distance as simple overlap measure
                if overlap > best_overlap:
                    best_overlap = overlap
                    closest_detection = detected
        
        if closest_detection is not None:
            total_comparisons += 1
            if closest_detection['predicted_damage'] == gt_damage:
                correct_predictions += 1
    
    # Create the 3-panel visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Original satellite image
    axes[0].imshow(img_original)
    axes[0].set_title('Original Satellite Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Ground truth
    axes[1].imshow(img_ground_truth)
    axes[1].set_title('Ground Truth Damage Classification', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Panel 3: End-to-end predictions (localization + damage classification)
    axes[2].imshow(img_prediction)
    accuracy = correct_predictions / total_comparisons if total_comparisons > 0 else 0
    detection_count = len(detected_with_damage)
    gt_count = len(gt_buildings)
    axes[2].set_title(f'End-to-End Pipeline\nDetected: {detection_count} | GT: {gt_count}\nAccuracy: {accuracy:.1%} ({correct_predictions}/{total_comparisons})', 
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add color legend
    legend_elements = []
    for damage_type, color in colors.items():
        rgb_color = tuple(c/255.0 for c in color)
        legend_elements.append(plt.Rectangle((0,0),1,1, fc=rgb_color, label=damage_type.title()))
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
              bbox_to_anchor=(0.5, -0.05), fontsize=12)
    
    plt.tight_layout()
    
    # Save the visualization
    base_name = os.path.basename(image_path).replace('.png', '')
    output_path = os.path.join(output_dir, f'{base_name}_damage_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'image_name': base_name,
        'gt_buildings': len(gt_buildings),
        'detected_buildings': len(detected_with_damage),
        'matched_comparisons': total_comparisons,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'output_path': output_path
    }

def run_visual_inference():
    print("Loading localization and damage classification models...")
    model_result = load_models()
    if model_result is None:
        return
    
    loc_model, damage_model, device = model_result
    
    # Create output directory
    output_dir = 'damage_inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process test images
    test_images_dir = 'Data/test/images'
    test_labels_dir = 'Data/test/labels'
    
    if not os.path.exists(test_images_dir) or not os.path.exists(test_labels_dir):
        print(f"Test directories not found: {test_images_dir} or {test_labels_dir}")
        return
    
    image_files = [f for f in os.listdir(test_images_dir) if f.endswith('.png')]
    post_disaster_images = [f for f in image_files if 'post_disaster' in f]
    
    if len(post_disaster_images) == 0:
        print("No post-disaster images found in test directory")
        return
    
    # Select 2 images per disaster type (6 disasters total = 12 images)
    disaster_types = [
        'socal-fire',
        'hurricane-harvey', 
        'hurricane-michael',
        'palu-tsunami',
        'midwest-flooding',
        'santa-rosa-wildfire'
    ]
    
    selected_images = []
    disaster_counts = {}
    
    for disaster in disaster_types:
        disaster_images = [f for f in post_disaster_images if disaster in f][:2]
        selected_images.extend(disaster_images)
        disaster_counts[disaster] = len(disaster_images)
        print(f"  {disaster}: {len(disaster_images)} images selected")
    
    print(f"\nSelected {len(selected_images)} images total (2 per disaster):")
    for disaster, count in disaster_counts.items():
        print(f"  {disaster}: {count} images")
    print("Processing images...")
    
    results = []
    total_accuracy = 0
    processed_count = 0
    
    # Process selected images
    for i, img_file in enumerate(selected_images):
        # Determine disaster type
        disaster_type = None
        for disaster in disaster_types:
            if disaster in img_file:
                disaster_type = disaster
                break
        
        print(f"Processing {i+1}/{len(selected_images)}: {img_file} ({disaster_type})")
        
        image_path = os.path.join(test_images_dir, img_file)
        label_file = img_file.replace('.png', '.json')
        label_path = os.path.join(test_labels_dir, label_file)
        
        if os.path.exists(label_path):
            result = process_image(image_path, label_path, loc_model, damage_model, device, output_dir)
            if result:
                results.append(result)
                result['disaster_type'] = disaster_type
                total_accuracy += result['accuracy']
                processed_count += 1
                print(f"  -> {result['accuracy']:.1%} accuracy ({result['correct_predictions']}/{result['matched_comparisons']} matched)")
                print(f"  -> Detected: {result['detected_buildings']} | GT: {result['gt_buildings']}")
                print(f"  -> Saved: {result['output_path']}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("DAMAGE CLASSIFICATION INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Processed images: {processed_count}")
    print(f"Average accuracy: {total_accuracy/processed_count:.1%}" if processed_count > 0 else "No images processed")
    print(f"Output directory: {output_dir}")
    
    print(f"\nDetailed Results by Disaster:")
    for disaster in disaster_types:
        disaster_results = [r for r in results if r.get('disaster_type') == disaster]
        if disaster_results:
            avg_accuracy = sum(r['accuracy'] for r in disaster_results) / len(disaster_results)
            print(f"\n{disaster.upper()} ({len(disaster_results)} images) - Avg: {avg_accuracy:.1%}:")
            for result in disaster_results:
                print(f"  {result['image_name']}: {result['accuracy']:.1%} ({result['correct_predictions']}/{result['matched_comparisons']}) | Detected: {result['detected_buildings']} | GT: {result['gt_buildings']}")
        else:
            print(f"\n{disaster.upper()}: No images processed")

if __name__ == '__main__':
    run_visual_inference() 