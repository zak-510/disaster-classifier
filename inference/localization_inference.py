import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.model import create_model
from data_processing.localization_data import XBDDataset
from tests.test_localization_inference import (
    load_localization_model, 
    load_test_image_and_labels, 
    predict_localization, 
    create_three_panel_visualization
)

def run_inference():
    print('LOCALIZATION MODEL INFERENCE')
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
    print('LOCALIZATION INFERENCE COMPLETE')
    print(f'Visualizations saved to {output_dir}/')

if __name__ == '__main__':
    run_inference() 