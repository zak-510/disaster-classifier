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
from tests.test_damage_inference import process_image, load_test_image_and_labels, load_models


def run_damage_inference():
    device = torch.device('cuda')
    print(f'DAMAGE CLASSIFIER INFERENCE')
    print('=' * 60)
    
    # Load the trained model
    model_path = os.path.join(project_root, 'weights/best_damage_model_optimized.pth')
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
    
    for image_id in test_images:
        print(f'\nProcessing Test Image: {image_id}')
        
        image, labels_data = load_test_image_and_labels(image_id)
        if image is None:
            print(f'  ERROR: Could not load image or labels for {image_id}')
            continue
            
        output_path = os.path.join(output_dir, f'damage_test_{test_images.index(image_id) + 1}.png')
        
        try:
            process_image(loc_model, damage_model, device, image, labels_data, image_id, output_path)
        except Exception as e:
            print(f'  ERROR: Failed to process {image_id}: {e}')
    
    print(f'\nSUCCESS: Test complete. Visualizations saved to {output_dir}/')

if __name__ == '__main__':
    run_damage_inference() 