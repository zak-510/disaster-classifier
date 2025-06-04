import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

"""
damage_classification.py
=======================
Trains a damage classification model on the xBD dataset.

Usage:
    python damage_classification.py --data_dir Data --output_dir xbd_pipeline_output/damage_model [--epochs 10] [--batch_size 4] [--test_mode]

Arguments:
    --data_dir: Path to raw xBD data directory (with train/images and train/labels).
    --output_dir: Path to save model checkpoints and logs.
    --epochs: (Optional) Number of training epochs (default: 10).
    --batch_size: (Optional) Batch size (default: 4).
    --test_mode: (Optional) If set, runs on a small subset for quick testing.
"""

class XBDDamageDataset(Dataset):
    def __init__(self, data_dir, transform=None, test_mode=False):
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all pre-disaster images
        self.pre_images = [f for f in os.listdir(os.path.join(data_dir, 'train', 'images')) 
                          if f.endswith('_pre_disaster.png')]
        if test_mode:
            self.pre_images = self.pre_images[:4]  # Use fewer images in test mode
            
    def __len__(self):
        return len(self.pre_images)
        
    def __getitem__(self, idx):
        pre_img_name = self.pre_images[idx]
        post_img_name = pre_img_name.replace('_pre_disaster', '_post_disaster')
        label_name = pre_img_name.replace('_pre_disaster.png', '_post_disaster.json')
        
        # Load pre and post disaster images
        pre_img = Image.open(os.path.join(self.data_dir, 'train', 'images', pre_img_name)).convert('RGB')
        post_img = Image.open(os.path.join(self.data_dir, 'train', 'images', post_img_name)).convert('RGB')
        
        # Load and parse JSON label
        with open(os.path.join(self.data_dir, 'train', 'labels', label_name), 'r') as f:
            label_data = json.load(f)
            
        # Get damage level (0-4) from the first building in the label
        # In a real implementation, you would process all buildings
        damage_level = 0  # Default to no damage
        if 'features' in label_data and 'xy' in label_data['features']:
            for building in label_data['features']['xy']:
                if 'properties' in building and 'damage' in building['properties']:
                    damage_level = building['properties']['damage']
                    break
        
        if self.transform:
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)
            
        # Stack pre and post images
        combined_img = torch.cat([pre_img, post_img], dim=0)
        return combined_img, damage_level

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser(description="Train damage classification model.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to raw xBD data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--test_mode', action='store_true', help='Run on a small subset for quick testing')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"[Info] Using device: {device}")

    # Create dataset and dataloader
    train_set = XBDDamageDataset(
        args.data_dir,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        test_mode=args.test_mode
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # Load pre-trained ResNet and modify for our task
    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 6 channels for pre/post
    model.fc = nn.Linear(model.fc.in_features, 5)  # 5 damage levels
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        accuracy = 100. * correct / total
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch{epoch+1}.pth"))
    print("[Done] Damage classification model training complete.")

if __name__ == "__main__":
    main() 