import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Directories
IMG_DIR = os.path.join('Data', 'test', 'images')
GT_MASK_DIR = os.path.join('Data', 'test', 'masks')
PRED_MASK_DIR = 'predictions'
DEBUG_DIR = os.path.join('predictions', 'debug_masks')

# Get list of pre-disaster images
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith('_pre_disaster.png')]

# Visualize first 5 images
for img_file in image_files[:5]:
    base = img_file.replace('_pre_disaster.png', '')
    
    # Load image
    img_path = os.path.join(IMG_DIR, img_file)
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load ground truth mask
    gt_mask_path = os.path.join(GT_MASK_DIR, base + '_mask.png')
    if os.path.exists(gt_mask_path):
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"Failed to load ground truth mask: {gt_mask_path}")
            continue
    else:
        print(f"Ground truth mask not found: {gt_mask_path}")
        continue
    
    # Load predicted mask
    pred_mask_path = os.path.join(PRED_MASK_DIR, base + '_mask.png')
    if os.path.exists(pred_mask_path):
        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            print(f"Failed to load predicted mask: {pred_mask_path}")
            continue
    else:
        print(f"Predicted mask not found: {pred_mask_path}")
        continue
    
    # Load overlay
    overlay_path = os.path.join(PRED_MASK_DIR, base + '_damage_overlay.png')
    if os.path.exists(overlay_path):
        overlay = cv2.imread(overlay_path)
        if overlay is None:
            print(f"Failed to load overlay: {overlay_path}")
            continue
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    else:
        print(f"Overlay not found: {overlay_path}")
        continue
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot original image
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot ground truth mask
    plt.subplot(2, 2, 2)
    plt.imshow(gt_mask, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    # Plot predicted mask
    plt.subplot(2, 2, 3)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    # Plot damage overlay
    plt.subplot(2, 2, 4)
    plt.imshow(overlay)
    plt.title('Damage Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show() 