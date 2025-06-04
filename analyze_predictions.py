import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch

def load_image(path):
    """Load image and convert to RGB."""
    return np.array(Image.open(path).convert('RGB'))

def load_mask(path):
    """Load mask as binary numpy array."""
    mask = np.array(Image.open(path).convert('L'))
    return (mask > 127).astype(np.uint8)

def compute_metrics(pred_mask, gt_mask):
    """Compute IoU, Dice coefficient, and pixel accuracy."""
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # Pixel accuracy
    accuracy = accuracy_score(gt_flat, pred_flat)
    
    # IoU (Intersection over Union)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0
    
    # Dice coefficient
    dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0
    
    # Precision and recall
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    
    return {
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

def create_overlay(image, mask, color=(0, 255, 0)):
    """Create proper overlay visualization."""
    overlay = image.copy()
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask == 1] = color
    cv2.addWeighted(mask_rgb, 0.3, overlay, 0.7, 0, overlay)
    return overlay

def analyze_predictions(data_dir, pred_dir, thresholds=[0.3, 0.5, 0.7]):
    """Analyze predictions for different thresholds."""
    # Get test images
    test_images = [f for f in os.listdir(os.path.join(data_dir, 'test', 'images')) 
                   if f.endswith('_pre_disaster.png')][:4]  # Analyze first 4 images
    
    for img_name in test_images:
        print(f"\nAnalyzing {img_name}...")
        
        # Load original image
        orig_img = load_image(os.path.join(data_dir, 'test', 'images', img_name))
        
        # Load ground truth mask
        gt_mask_path = os.path.join(data_dir, 'test', 'masks', 
                                   img_name.replace('_pre_disaster.png', '_mask.png'))
        gt_mask = load_mask(gt_mask_path)
        
        # Load probability mask
        prob_mask_path = os.path.join(pred_dir, img_name.replace('.png', '_prob_mask.png'))
        prob_mask = np.array(Image.open(prob_mask_path)) / 255.0
        
        # Create figure for this image
        plt.figure(figsize=(20, 10))
        
        # Plot original image
        plt.subplot(2, 3, 1)
        plt.imshow(orig_img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot ground truth mask
        plt.subplot(2, 3, 2)
        plt.imshow(gt_mask, cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')
        
        # Plot probability mask
        plt.subplot(2, 3, 3)
        plt.imshow(prob_mask, cmap='jet')
        plt.colorbar()
        plt.title('Probability Map')
        plt.axis('off')
        
        # Create table for metrics
        metrics_data = []
        headers = ['Threshold', 'IoU', 'Dice', 'Accuracy', 'Precision', 'Recall']
        
        for threshold in thresholds:
            # Create binary mask for this threshold
            pred_mask = (prob_mask > threshold).astype(np.uint8)
            
            # Compute metrics
            metrics = compute_metrics(pred_mask, gt_mask)
            metrics_data.append([
                f"{threshold:.1f}",
                f"{metrics['iou']:.3f}",
                f"{metrics['dice']:.3f}",
                f"{metrics['accuracy']:.3f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}"
            ])
            
            # Create and save overlay
            overlay = create_overlay(orig_img, pred_mask)
            overlay_path = os.path.join(pred_dir, 
                                      f"{img_name.replace('.png', f'_overlay_t{threshold:.1f}.png')}")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Plot binary mask and overlay for threshold 0.5
            if threshold == 0.5:
                plt.subplot(2, 3, 4)
                plt.imshow(pred_mask, cmap='gray')
                plt.title(f'Binary Mask (t={threshold:.1f})')
                plt.axis('off')
                
                plt.subplot(2, 3, 5)
                plt.imshow(overlay)
                plt.title(f'Overlay (t={threshold:.1f})')
                plt.axis('off')
        
        # Add metrics table
        plt.subplot(2, 3, 6)
        plt.axis('off')
        table = plt.table(cellText=metrics_data, colLabels=headers, 
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        plt.title('Metrics Comparison')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(pred_dir, f"{img_name.replace('.png', '_analysis.png')}"))
        plt.close()
        
        # Print metrics
        print("\nMetrics for different thresholds:")
        print("Threshold | IoU     | Dice    | Accuracy | Precision | Recall")
        print("-" * 65)
        for row in metrics_data:
            print(f"{row[0]:9} | {row[1]:7} | {row[2]:7} | {row[3]:8} | {row[4]:9} | {row[5]:6}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Analyze model predictions.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to raw xBD data directory')
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to predictions directory')
    args = parser.parse_args()
    
    analyze_predictions(args.data_dir, args.pred_dir) 