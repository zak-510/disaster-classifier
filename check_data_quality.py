import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import json
from train_localization import XBDDataset, parse_wkt_polygon
import cv2
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_image_stats(image_path):
    """Analyze basic statistics of an image."""
    img = np.array(Image.open(image_path).convert('RGB'))
    return {
        'shape': img.shape,
        'dtype': img.dtype,
        'min': img.min(),
        'max': img.max(),
        'mean': img.mean(),
        'std': img.std()
    }

def check_mask_quality(mask):
    """Check quality of a binary mask."""
    return {
        'shape': mask.shape,
        'dtype': mask.dtype,
        'unique_values': np.unique(mask),
        'min': mask.min(),
        'max': mask.max(),
        'mean': mask.mean(),
        'building_pixels': np.sum(mask > 0),
        'total_pixels': mask.size,
        'building_ratio': np.sum(mask > 0) / mask.size
    }

def visualize_sample(pre_img, post_img, mask, save_path):
    """Visualize a sample with pre/post images and mask."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(pre_img)
    plt.title('Pre-disaster Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(post_img)
    plt.title('Post-disaster Image')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(mask, cmap='gray')
    plt.title('Building Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_dataset(data_dir, output_dir, num_samples=5):
    """Analyze dataset quality and preprocessing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset
    dataset = XBDDataset(data_dir, split='train', image_size=(512, 512))
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Analyze data distribution
    building_ratios = []
    image_stats = []
    mask_stats = []
    
    for i in tqdm(range(min(len(dataset), num_samples)), desc="Analyzing samples"):
        try:
            # Get sample
            sample = dataset[i]
            pre_img = sample['pre_image']
            post_img = sample['post_image']
            mask = sample['mask']
            disaster_id = sample['disaster_id']
            
            logger.info(f"Processing sample {i}: {disaster_id}")
            
            # Convert tensors to numpy arrays for visualization
            pre_img = pre_img.numpy().transpose(1, 2, 0)
            post_img = post_img.numpy().transpose(1, 2, 0)
            mask = mask.numpy().squeeze()  # Remove channel dimension
            
            # Analyze mask quality
            mask_quality = check_mask_quality(mask)
            mask_stats.append(mask_quality)
            building_ratios.append(mask_quality['building_ratio'])
            
            # Analyze image statistics
            img_stats = {
                'pre_img': {
                    'shape': pre_img.shape,
                    'dtype': pre_img.dtype,
                    'min': pre_img.min(),
                    'max': pre_img.max(),
                    'mean': pre_img.mean(),
                    'std': pre_img.std()
                },
                'post_img': {
                    'shape': post_img.shape,
                    'dtype': post_img.dtype,
                    'min': post_img.min(),
                    'max': post_img.max(),
                    'mean': post_img.mean(),
                    'std': post_img.std()
                }
            }
            image_stats.append(img_stats)
            
            # Visualize sample
            save_path = os.path.join(output_dir, f'sample_{i}_{disaster_id}.png')
            visualize_sample(pre_img, post_img, mask, save_path)
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Print summary statistics
    logger.info("\nDataset Summary:")
    logger.info(f"Number of samples analyzed: {len(building_ratios)}")
    
    if building_ratios:
        logger.info(f"Average building ratio: {np.mean(building_ratios):.3f}")
        logger.info(f"Building ratio std: {np.std(building_ratios):.3f}")
    
        # Check for potential issues
        issues = []
        
        # Check image normalization
        for i, stats in enumerate(image_stats):
            if stats['pre_img']['max'] > 1.0 or stats['post_img']['max'] > 1.0:
                issues.append(f"Sample {i}: Images not normalized to [0,1]")
            if abs(stats['pre_img']['mean']) < 1e-6 or abs(stats['post_img']['mean']) < 1e-6:
                issues.append(f"Sample {i}: Potential zero-value images")
                
        # Check mask issues
        for i, stats in enumerate(mask_stats):
            if not np.all(np.isin(stats['unique_values'], [0, 1])):
                issues.append(f"Sample {i}: Mask contains values other than 0 and 1")
            if stats['building_ratio'] < 0.01:
                issues.append(f"Sample {i}: Very few building pixels ({stats['building_ratio']:.3%})")
        
        if issues:
            logger.warning("\nPotential Issues Found:")
            for issue in issues:
                logger.warning(f"- {issue}")
        else:
            logger.info("\nNo major issues found in analyzed samples.")
    else:
        logger.warning("\nNo samples were successfully analyzed!")
    
    return {
        'building_ratios': building_ratios,
        'image_stats': image_stats,
        'mask_stats': mask_stats,
        'issues': issues if building_ratios else []
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Check data quality and preprocessing.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save analysis results')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to analyze')
    args = parser.parse_args()
    
    analyze_dataset(args.data_dir, args.output_dir, args.num_samples) 