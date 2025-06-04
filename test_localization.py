import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # Initialize with smaller weights to prevent explosion
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Encoder for pre-disaster images
        self.pre_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 256x256
        )
        
        # Encoder for post-disaster images
        self.post_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 256x256
        )
        
        # Combined processing
        self.middle = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 128x128
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # 128x128 -> 256x256
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 256x256 -> 512x512
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)  # Final 1x1 conv to get single channel
        )
        
        # Initialize weights
        self.apply(init_weights)

    def validate_input(self, pre_img, post_img):
        """Validate input shapes and types."""
        if not isinstance(pre_img, torch.Tensor) or not isinstance(post_img, torch.Tensor):
            raise ValueError("Inputs must be torch tensors")
            
        if pre_img.dim() != 4 or post_img.dim() != 4:
            raise ValueError(f"Inputs must be 4D tensors (batch, channels, height, width), got shapes {pre_img.shape} and {post_img.shape}")
            
        if pre_img.shape != post_img.shape:
            raise ValueError(f"Pre and post disaster images must have same shape, got {pre_img.shape} and {post_img.shape}")
            
        if pre_img.shape[1] != 3:
            raise ValueError(f"Input images must have 3 channels, got {pre_img.shape[1]}")

    def forward(self, pre_img, post_img):
        """
        Forward pass
        Args:
            pre_img: Pre-disaster image tensor (B, 3, H, W)
            post_img: Post-disaster image tensor (B, 3, H, W)
        Returns:
            Building mask logits tensor (B, 1, H, W)
        """
        # Validate inputs
        self.validate_input(pre_img, post_img)
        
        # Normalize inputs to [-1, 1]
        pre_img = (pre_img - 0.5) * 2
        post_img = (post_img - 0.5) * 2
        
        # Encode both images
        pre_features = self.pre_encoder(pre_img)    # B, 32, H/2, W/2
        post_features = self.post_encoder(post_img)  # B, 32, H/2, W/2
        
        # Concatenate features
        combined = torch.cat([pre_features, post_features], dim=1)  # B, 64, H/2, W/2
        
        # Process combined features
        middle = self.middle(combined)  # B, 64, H/4, W/4
        
        # Decode to logits (no sigmoid activation)
        output = self.decoder(middle)  # B, 1, H, W
        
        return output

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
        post_img_name = pre_img_name.replace('_pre_disaster.png', '_post_disaster.png')
        
        # Load pre and post disaster images
        pre_img = Image.open(os.path.join(self.data_dir, 'test', 'images', pre_img_name)).convert('RGB')
        post_img = Image.open(os.path.join(self.data_dir, 'test', 'images', post_img_name)).convert('RGB')
        
        if self.transform:
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)
            
        return pre_img, post_img, pre_img_name

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test localization model.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to raw xBD data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--model_path', type=str, required=True, help='Path to localization model checkpoint')
    parser.add_argument('--test_mode', action='store_true', help='Run on a small subset for quick testing')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Using device: {device}")

    # Create dataset and dataloader
    test_set = XBDTestDataset(
        args.data_dir,
        transform=transforms.Compose([
            transforms.Resize((512, 512)),  # Match training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        test_mode=args.test_mode
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Load model
    print("[Info] Loading model...")
    model = SimpleUNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Run inference
    print("[Info] Running inference...")
    with torch.no_grad():
        for pre_imgs, post_imgs, img_names in tqdm(test_loader, desc="Processing images"):
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)
            
            # Get predictions
            outputs = model(pre_imgs, post_imgs)
            
            # Process each image
            for i, img_name in enumerate(img_names):
                # Load original image to get dimensions
                orig_img_path = os.path.join(args.data_dir, 'test', 'images', img_name)
                orig_img = np.array(Image.open(orig_img_path).convert('RGB'))
                h, w = orig_img.shape[:2]
                
                # Convert output to mask
                mask = torch.sigmoid(outputs[i]).cpu().numpy()
                mask = np.squeeze(mask)  # Remove channel dimension
                
                # Resize mask to original image size
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Save raw probability mask
                prob_mask_path = os.path.join(args.output_dir, f"{img_name.replace('.png', '_prob_mask.png')}")
                cv2.imwrite(prob_mask_path, (mask_resized * 255).astype(np.uint8))
                
                # Create binary mask
                binary_mask = (mask_resized > 0.5).astype(np.uint8)
                binary_mask_path = os.path.join(args.output_dir, f"{img_name.replace('.png', '_binary_mask.png')}")
                cv2.imwrite(binary_mask_path, binary_mask * 255)
                
                # Create overlay visualization
                overlay = orig_img.copy()
                overlay[binary_mask == 1] = overlay[binary_mask == 1] * 0.7 + np.array([0, 255, 0], dtype=np.uint8) * 0.3
                overlay_path = os.path.join(args.output_dir, f"{img_name.replace('.png', '_overlay.png')}")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("[Done] Inference completed. Results saved to:", args.output_dir)

if __name__ == '__main__':
    main() 