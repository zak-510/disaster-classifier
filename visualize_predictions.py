import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
from pathlib import Path
from train_localization import SimpleUNet

def load_model(model_path, device):
    """Load the trained model."""
    model = SimpleUNet()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def load_image(image_path):
    """Load and preprocess an image."""
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    return transform(image)

def visualize_sample(pre_img, post_img, true_mask, pred_mask, output_path):
    """Create a visualization grid showing inputs, ground truth and prediction."""
    plt.figure(figsize=(15, 5))
    
    # Pre-disaster image
    plt.subplot(151)
    plt.imshow(pre_img)
    plt.title('Pre-disaster')
    plt.axis('off')
    
    # Post-disaster image
    plt.subplot(152)
    plt.imshow(post_img)
    plt.title('Post-disaster')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(153)
    plt.imshow(true_mask, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Predicted mask
    plt.subplot(154)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    
    # Overlay prediction on pre-disaster image
    plt.subplot(155)
    overlay = pre_img.copy()
    overlay[pred_mask > 0.5] = [255, 0, 0]  # Red overlay for predictions
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, device)
    
    # Get validation image paths
    val_dir = os.path.join(args.data_dir, 'val')
    images_dir = os.path.join(val_dir, 'images')
    masks_dir = os.path.join(val_dir, 'masks')
    
    pre_images = [f for f in os.listdir(images_dir) if f.endswith('_pre_disaster.png')]
    print(f"Found {len(pre_images)} validation images")
    
    # Run inference on samples
    print(f"Generating visualizations for {args.num_samples} samples...")
    for i, pre_img_name in enumerate(pre_images[:args.num_samples]):
        try:
            # Get corresponding file names
            base_name = pre_img_name.replace('_pre_disaster.png', '')
            post_img_name = f"{base_name}_post_disaster.png"
            mask_name = f"{base_name}_mask.png"
            
            # Load images
            pre_img = load_image(os.path.join(images_dir, pre_img_name))
            post_img = load_image(os.path.join(images_dir, post_img_name))
            true_mask = Image.open(os.path.join(masks_dir, mask_name))
            true_mask = transforms.Resize((512, 512))(transforms.ToTensor()(true_mask))
            
            # Generate prediction
            with torch.no_grad():
                pre_img_input = pre_img.unsqueeze(0).to(device)
                post_img_input = post_img.unsqueeze(0).to(device)
                pred_mask = model(pre_img_input, post_img_input)
                pred_mask = torch.sigmoid(pred_mask)
            
            # Convert tensors to numpy for visualization
            pre_img = pre_img.permute(1, 2, 0).cpu().numpy()
            post_img = post_img.permute(1, 2, 0).cpu().numpy()
            true_mask = true_mask[0].cpu().numpy()
            pred_mask = pred_mask[0, 0].cpu().numpy()
            
            # Normalize images for display
            pre_img = (pre_img * 255).astype(np.uint8)
            post_img = (post_img * 255).astype(np.uint8)
            
            # Save visualization
            output_path = os.path.join(args.output_dir, f'prediction_{base_name}.png')
            visualize_sample(pre_img, post_img, true_mask, pred_mask, output_path)
            print(f"Saved visualization to {output_path}")
            
        except Exception as e:
            print(f"Error processing {pre_img_name}: {str(e)}")
            continue
    
    print("Done!")

if __name__ == '__main__':
    main() 