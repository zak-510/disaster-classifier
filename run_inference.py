import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import cv2

"""
run_inference.py
===============
Runs inference on the test set using trained localization and damage classification models.

Usage:
    python run_inference.py --data_dir Data --output_dir xbd_pipeline_output/predictions --localization_model xbd_pipeline_output/localization_model/model_epoch10.pth --damage_model xbd_pipeline_output/damage_model/model_epoch10.pth [--test_mode]

Arguments:
    --data_dir: Path to raw xBD data directory (with test/images and test/labels).
    --output_dir: Path to save predictions.
    --localization_model: Path to trained localization model checkpoint.
    --damage_model: Path to trained damage classification model checkpoint.
    --test_mode: (Optional) If set, runs on a small subset for quick testing.
"""

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class XBDTestDataset(Dataset):
    def __init__(self, data_dir, transform=None, test_mode=False):
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all pre-disaster images
        self.pre_images = [f for f in os.listdir(os.path.join(data_dir, 'test', 'images')) 
                          if f.endswith('_pre_disaster.png')]
        if test_mode:
            self.pre_images = self.pre_images[:4]  # Use fewer images in test mode
            
    def __len__(self):
        return len(self.pre_images)
        
    def __getitem__(self, idx):
        pre_img_name = self.pre_images[idx]
        post_img_name = pre_img_name.replace('_pre_disaster', '_post_disaster')
        
        # Load pre and post disaster images
        pre_img = Image.open(os.path.join(self.data_dir, 'test', 'images', pre_img_name)).convert('RGB')
        post_img = Image.open(os.path.join(self.data_dir, 'test', 'images', post_img_name)).convert('RGB')
        
        if self.transform:
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)
            
        return pre_img, post_img, pre_img_name

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_localization_model(model_path, device):
    model = SimpleUNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_damage_model(model_path, device):
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def process_predictions(localization_output, damage_output, threshold=0.1, img_name=None, orig_shape=None):
    # Convert localization output to mask
    mask = torch.sigmoid(localization_output).cpu().numpy()
    # Save raw mask for debugging
    if img_name is not None and orig_shape is not None:
        raw_mask = cv2.resize(mask[0,0], (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f"raw_mask_{img_name}.png", (raw_mask * 255).astype(np.uint8))
    # Threshold
    binary_mask = (mask > threshold).astype(np.uint8)
    # Get damage predictions
    damage_probs = torch.softmax(damage_output, dim=1).cpu().numpy()
    damage_levels = np.argmax(damage_probs, axis=1)
    return binary_mask, damage_levels

def get_color_for_damage(damage_level):
    # Color mapping for damage levels (BGR for OpenCV)
    colors = {
        0: (0, 255, 0),      # Green: No damage
        1: (0, 255, 255),    # Yellow: Minor damage
        2: (0, 165, 255),    # Orange: Major damage
        3: (0, 0, 255),      # Red: Destroyed
        4: (255, 0, 0)       # Blue: Unclassified
    }
    return colors.get(damage_level, (128, 128, 128))

def create_instance_overlay(
    pre_img_path, post_img_path, mask, damage_model, device, alpha=0.5, debug_dir=None, img_name=None
):
    # Ensure mask is binary and matches original image size
    mask = (mask > 0.5).astype(np.uint8)
    pre_img_np = np.array(Image.open(pre_img_path).convert('RGB'))
    post_img_np = np.array(Image.open(post_img_path).convert('RGB'))
    h, w = pre_img_np.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    print(f"[DEBUG] {img_name}: mask shape {mask.shape}, unique values {np.unique(mask)}")
    # Save binary mask for inspection
    if debug_dir and img_name:
        cv2.imwrite(os.path.join(debug_dir, f"binary_mask_{img_name}.png"), (mask * 255).astype(np.uint8))
    if np.sum(mask) == 0:
        print(f"[WARNING] {img_name}: Binary mask is empty after thresholding and resizing.")
    num_labels, labels = cv2.connectedComponents(mask)
    overlay = pre_img_np.copy()
    if debug_dir and img_name:
        os.makedirs(debug_dir, exist_ok=True)
    for label in range(1, num_labels):
        building_mask = (labels == label)
        if not np.any(building_mask):
            continue
        # Save per-building mask (white on black)
        if debug_dir and img_name:
            mask_img = (building_mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(debug_dir, f"{img_name}_building_{label}.png"), mask_img)
        # Crop bounding box from original images
        ys, xs = np.where(building_mask)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        pre_crop = pre_img_np[y1:y2+1, x1:x2+1, :]
        post_crop = post_img_np[y1:y2+1, x1:x2+1, :]
        # Save crop for debug
        if debug_dir and img_name:
            cv2.imwrite(os.path.join(debug_dir, f"{img_name}_building_{label}_pre_crop.png"), cv2.cvtColor(pre_crop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(debug_dir, f"{img_name}_building_{label}_post_crop.png"), cv2.cvtColor(post_crop, cv2.COLOR_RGB2BGR))
        # Prepare crop for classifier
        pre_crop_tensor = transforms.ToTensor()(Image.fromarray(pre_crop)).unsqueeze(0)
        post_crop_tensor = transforms.ToTensor()(Image.fromarray(post_crop)).unsqueeze(0)
        pre_crop_resized = torch.nn.functional.interpolate(pre_crop_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        post_crop_resized = torch.nn.functional.interpolate(post_crop_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        combined = torch.cat([pre_crop_resized, post_crop_resized], dim=1)
        with torch.no_grad():
            damage_out = damage_model(combined.to(device))
            damage_level = torch.argmax(torch.softmax(damage_out, dim=1), dim=1).item()
        color = np.array(get_color_for_damage(damage_level), dtype=np.uint8)
        # Alpha blend only on the building mask
        for c in range(3):
            overlay[..., c] = np.where(
                building_mask,
                (1 - alpha) * overlay[..., c] + alpha * color[c],
                overlay[..., c]
            ).astype(np.uint8)
    return overlay

def save_predictions(output_dir, image_name, mask, damage_levels):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Squeeze mask to ensure it's 2D
    mask = np.squeeze(mask)
    
    # Save binary mask
    mask_path = os.path.join(output_dir, f"{image_name.replace('_pre_disaster.png', '_mask.png')}")
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
    
    # Create and save damage visualization
    damage_vis = create_damage_visualization(mask, damage_levels)
    damage_vis_path = os.path.join(output_dir, f"{image_name.replace('_pre_disaster.png', '_damage_vis.png')}")
    cv2.imwrite(damage_vis_path, damage_vis)
    
    # Save damage levels as JSON
    damage_path = os.path.join(output_dir, f"{image_name.replace('_pre_disaster.png', '_damage.json')}")
    with open(damage_path, 'w') as f:
        json.dump({
            'damage_levels': damage_levels.tolist(),
            'image_name': image_name
        }, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run inference on test set.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to raw xBD data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--localization_model', type=str, required=True, help='Path to localization model checkpoint')
    parser.add_argument('--damage_model', type=str, required=True, help='Path to damage model checkpoint')
    parser.add_argument('--test_mode', action='store_true', help='Run on a small subset for quick testing')
    args = parser.parse_args()

    device = get_device()
    print(f"[Info] Using device: {device}")

    # Create dataset and dataloader
    test_set = XBDTestDataset(
        args.data_dir,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        test_mode=args.test_mode
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Load models
    print("[Info] Loading models...")
    localization_model = load_localization_model(args.localization_model, device)
    damage_model = load_damage_model(args.damage_model, device)

    # Run inference
    print("[Info] Running inference...")
    with torch.no_grad():
        debug_dir = os.path.join(args.output_dir, 'debug_masks')
        for pre_imgs, post_imgs, img_names in tqdm(test_loader, desc="Processing images"):
            pre_imgs, post_imgs = pre_imgs.to(device), post_imgs.to(device)
            
            # Get localization predictions
            localization_output = localization_model(pre_imgs)
            
            # Get damage predictions
            combined_imgs = torch.cat([pre_imgs, post_imgs], dim=1).to(device)
            damage_output = damage_model(combined_imgs)
            
            # Process predictions
            for i, img_name in enumerate(img_names):
                pre_img_path = os.path.join(args.data_dir, 'test', 'images', img_name)
                post_img_path = pre_img_path.replace('_pre_disaster.png', '_post_disaster.png')
                orig_img = np.array(Image.open(pre_img_path).convert('RGB'))
                h, w = orig_img.shape[:2]
                # Save and resize mask
                mask, damage_levels = process_predictions(localization_output, damage_output, threshold=0.1, img_name=img_name.replace('.png',''), orig_shape=(h, w))
                mask_resized = cv2.resize(np.squeeze(mask[i]), (w, h), interpolation=cv2.INTER_NEAREST)
                # Save raw mask for inspection
                cv2.imwrite(f"raw_mask_{img_name}.png", (orig_img * 255).astype(np.uint8))
                # Save binary mask
                mask_path = os.path.join(args.output_dir, f"{img_name.replace('_pre_disaster.png', '_mask.png')}")
                cv2.imwrite(mask_path, (mask_resized * 255).astype(np.uint8))
                # Overlay with alpha blending and debug mask saving
                overlay = create_instance_overlay(
                    pre_img_path, post_img_path, mask_resized, damage_model, device, alpha=0.5, debug_dir=debug_dir, img_name=img_name.replace('.png', '')
                )
                overlay_path = os.path.join(args.output_dir, f"{img_name.replace('_pre_disaster.png', '_damage_overlay.png')}")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                # Save JSON (for now, just save the mask-level damage, can be extended to per-building)
                damage_path = os.path.join(args.output_dir, f"{img_name.replace('_pre_disaster.png', '_damage.json')}")
                with open(damage_path, 'w') as f:
                    json.dump({
                        'image_name': img_name
                    }, f, indent=2)

    print(f"[Success] Predictions saved to {args.output_dir}")

if __name__ == "__main__":
    main() 