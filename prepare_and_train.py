import os
import shutil
import random
import argparse
from pathlib import Path
import subprocess

def split_data(data_dir, output_dir, val_ratio=0.2, test_mode=False):
    """Split data into train and validation sets."""
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    for split_dir in [train_dir, val_dir]:
        for subdir in ['images', 'labels', 'masks']:
            os.makedirs(os.path.join(split_dir, subdir), exist_ok=True)
    
    # Get all pre-disaster images
    images_dir = os.path.join(data_dir, 'images')
    pre_images = [f for f in os.listdir(images_dir) if f.endswith('_pre_disaster.png')]
    
    if test_mode:
        pre_images = pre_images[:10]  # Use only 10 images for testing
        
    # Randomly split images
    random.shuffle(pre_images)
    split_idx = int(len(pre_images) * (1 - val_ratio))
    train_images = pre_images[:split_idx]
    val_images = pre_images[split_idx:]
    
    # Copy files to respective directories
    for split, image_list in [('train', train_images), ('val', val_images)]:
        split_dir = os.path.join(output_dir, split)
        print(f"Copying {len(image_list)} images to {split} set...")
        
        for pre_img in image_list:
            # Get corresponding files
            disaster_id = pre_img.replace('_pre_disaster.png', '')
            post_img = f"{disaster_id}_post_disaster.png"
            label_file = f"{disaster_id}_pre_disaster.json"
            mask_file = f"{disaster_id}_mask.png"
            
            # Copy images
            shutil.copy2(
                os.path.join(data_dir, 'images', pre_img),
                os.path.join(split_dir, 'images', pre_img)
            )
            shutil.copy2(
                os.path.join(data_dir, 'images', post_img),
                os.path.join(split_dir, 'images', post_img)
            )
            
            # Copy label if exists
            label_src = os.path.join(data_dir, 'labels', label_file)
            if os.path.exists(label_src):
                shutil.copy2(
                    label_src,
                    os.path.join(split_dir, 'labels', label_file)
                )
                
            # Copy mask if exists
            mask_src = os.path.join(data_dir, 'masks', mask_file)
            if os.path.exists(mask_src):
                shutil.copy2(
                    mask_src,
                    os.path.join(split_dir, 'masks', mask_file)
                )
    
    return train_dir, val_dir

def main():
    parser = argparse.ArgumentParser(description="Prepare data and train localization model")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to raw data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    
    # Prepare data
    print("Preparing data...")
    train_dir, val_dir = split_data(
        args.data_dir,
        args.output_dir,
        val_ratio=args.val_ratio,
        test_mode=args.test_mode
    )
    
    # Run training
    print("\nStarting training...")
    train_cmd = [
        'python', 'train_localization.py',
        '--data_dir', args.output_dir,
        '--output_dir', os.path.join(args.output_dir, 'model'),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size)
    ]
    if args.test_mode:
        train_cmd.append('--test_mode')
        
    subprocess.run(train_cmd)

if __name__ == "__main__":
    main() 