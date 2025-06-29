import torch
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.damage_model import create_damage_model, calculate_accuracy
from data_processing.damage_data import get_damage_data_loaders
from tests.test_damage_inference import process_image, load_test_image_and_labels, load_models, get_ground_truth_buildings, detect_buildings, predict_damage
import cv2


def match_predictions_to_ground_truth(predicted_regions, predicted_damages, ground_truth_buildings):
    """Match predicted buildings to ground truth buildings and return matched pairs"""
    matched_predictions = []
    matched_ground_truth = []
    
    for pred_region, pred_damage in zip(predicted_regions, predicted_damages):
        pred_bbox = pred_region['bbox']
        pred_minr, pred_minc, pred_maxr, pred_maxc = pred_bbox
        
        best_overlap = 0
        best_gt_damage = None
        
        # Find the ground truth building with the highest overlap
        for gt_building in ground_truth_buildings:
            gt_bbox = gt_building['bbox']
            gt_minr, gt_minc, gt_maxr, gt_maxc = gt_bbox
            
            # Calculate intersection over union (IoU)
            inter_minr = max(pred_minr, gt_minr)
            inter_minc = max(pred_minc, gt_minc)
            inter_maxr = min(pred_maxr, gt_maxr)
            inter_maxc = min(pred_maxc, gt_maxc)
            
            if inter_minr < inter_maxr and inter_minc < inter_maxc:
                inter_area = (inter_maxr - inter_minr) * (inter_maxc - inter_minc)
                pred_area = (pred_maxr - pred_minr) * (pred_maxc - pred_minc)
                gt_area = (gt_maxr - gt_minr) * (gt_maxc - gt_minc)
                union_area = pred_area + gt_area - inter_area
                
                if union_area > 0:
                    overlap = inter_area / union_area
                    if overlap > best_overlap and overlap > 0.3:  # Minimum overlap threshold
                        best_overlap = overlap
                        best_gt_damage = gt_building['damage_type']
        
        if best_gt_damage is not None:
            matched_predictions.append(pred_damage)
            matched_ground_truth.append(best_gt_damage)
    
    return matched_predictions, matched_ground_truth


def run_damage_inference_with_f1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'DAMAGE CLASSIFIER INFERENCE WITH F1 SCORE')
    print('=' * 60)
    
    # Load the trained model
    model_path = os.path.join(project_root, 'models/weights/best_damage_model_optimized.pth')
    if not os.path.exists(model_path):
        print(f'ERROR: Model file {model_path} not found!')
        return
    
    # loc_model, damage_model, device
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
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(project_root, 'test_results/damage')
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all predictions and ground truth for F1 score calculation
    all_predictions = []
    all_ground_truth = []
    class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
    
    for image_id in test_images:
        print(f'\nProcessing Test Image: {image_id}')
        
        image, labels_data = load_test_image_and_labels(image_id)
        if image is None:
            print(f'  ERROR: Could not load image or labels for {image_id}')
            continue
            
        try:
            # Get ground truth buildings
            ground_truth_buildings = get_ground_truth_buildings(labels_data, image.shape)
            
            # Detect buildings using localization model
            predicted_building_regions = detect_buildings(loc_model, device, image)
            
            if len(predicted_building_regions) == 0:
                print(f'  WARNING: No buildings detected in {image_id}')
                continue
            
            # Predict damage for each detected building
            predicted_damages = []
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
            
            # Match predictions to ground truth
            matched_preds, matched_gt = match_predictions_to_ground_truth(
                predicted_building_regions, predicted_damages, ground_truth_buildings
            )
            
            # Add to overall collections
            all_predictions.extend(matched_preds)
            all_ground_truth.extend(matched_gt)
            
            print(f'  Processed: {len(ground_truth_buildings)} GT buildings, {len(predicted_building_regions)} detected, {len(matched_preds)} matched')
            
            # Create visualization
            output_path = os.path.join(output_dir, f'damage_test_{test_images.index(image_id) + 1}.png')
            process_image(loc_model, damage_model, device, image, labels_data, image_id, output_path)
            
        except Exception as e:
            print(f'  ERROR: Failed to process {image_id}: {e}')
    
    # Calculate and display F1 scores
    if len(all_predictions) > 0 and len(all_ground_truth) > 0:
        print(f'\n' + '='*60)
        print(f'F1 SCORE RESULTS')
        print(f'='*60)
        print(f'Total matched building pairs: {len(all_predictions)}')
        
        # Overall F1 scores
        f1_macro = f1_score(all_ground_truth, all_predictions, average='macro')
        f1_micro = f1_score(all_ground_truth, all_predictions, average='micro')
        f1_weighted = f1_score(all_ground_truth, all_predictions, average='weighted')
        
        print(f'\nOverall F1 Scores:')
        print(f'  Macro F1:    {f1_macro:.3f}')
        print(f'  Micro F1:    {f1_micro:.3f}')
        print(f'  Weighted F1: {f1_weighted:.3f}')
        
        # Per-class F1 scores
        f1_per_class = f1_score(all_ground_truth, all_predictions, average=None)
        print(f'\nPer-Class F1 Scores:')
        for i, (class_name, f1_val) in enumerate(zip(class_names, f1_per_class)):
            print(f'  {class_name:15}: {f1_val:.3f}')
        
        # Additional metrics
        accuracy = accuracy_score(all_ground_truth, all_predictions)
        precision_macro = precision_score(all_ground_truth, all_predictions, average='macro')
        recall_macro = recall_score(all_ground_truth, all_predictions, average='macro')
        
        print(f'\nAdditional Metrics:')
        print(f'  Accuracy:           {accuracy:.3f}')
        print(f'  Precision (macro):  {precision_macro:.3f}')
        print(f'  Recall (macro):     {recall_macro:.3f}')
        
        # Confusion matrix
        cm = confusion_matrix(all_ground_truth, all_predictions)
        print(f'\nConfusion Matrix:')
        print(f'{"":15} {"Predicted":>40}')
        print(f'{"GT / Actual":15} {" ".join(f"{name:10}" for name in class_names)}')
        for i, class_name in enumerate(class_names):
            print(f'{class_name:15} {" ".join(f"{cm[i,j]:10d}" for j in range(len(class_names)))}')
        
        # Classification report
        print(f'\nDetailed Classification Report:')
        print(classification_report(all_ground_truth, all_predictions, target_names=class_names))
        
    else:
        print(f'\nWARNING: No matched predictions found for F1 score calculation')
    
    print(f'\nSUCCESS: Test complete. Visualizations saved to {output_dir}/')
    return f1_macro if len(all_predictions) > 0 else None


def run_damage_inference():
    """Original function for backward compatibility"""
    return run_damage_inference_with_f1()


if __name__ == '__main__':
    run_damage_inference_with_f1() 