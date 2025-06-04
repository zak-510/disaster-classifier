import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import time
import yaml
import psutil
from pathlib import Path
import GPUtil
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
import logging
import cv2

"""
train_localization.py
====================
Trains a building localization (segmentation) model on the xBD dataset.

Usage:
    python train_localization.py --data_dir Data --output_dir xbd_pipeline_output/localization_model [--epochs 10] [--batch_size 4] [--test_mode]

Arguments:
    --data_dir: Path to raw xBD data directory (with images and labels).
    --output_dir: Path to save model checkpoints and logs.
    --epochs: (Optional) Number of training epochs (default: 10).
    --batch_size: (Optional) Batch size (default: 4).
    --test_mode: (Optional) If set, runs on a small subset for quick testing.

Example:
    python train_localization.py --data_dir Data --output_dir xbd_pipeline_output/localization_model
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Timer:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.start_time = time.time()
        self.splits = {}
        
    def split(self, name):
        current = time.time()
        duration = current - self.start_time
        self.splits[name] = duration
        self.start_time = current
        return duration

def validate_data_paths(config):
    """Validate that all required data directories and files exist."""
    base_dir = config['data']['base_dir']
    train_dir = os.path.join(base_dir, config['data']['train_dir'])
    test_dir = os.path.join(base_dir, config['data']['test_dir'])
    
    required_dirs = {
        'base_dir': base_dir,
        'train_dir': train_dir,
        'train_images': os.path.join(train_dir, config['data']['image_dir']),
        'train_labels': os.path.join(train_dir, config['data']['label_dir']),
        'train_masks': os.path.join(train_dir, config['data']['mask_dir']),
        'test_dir': test_dir,
        'test_images': os.path.join(test_dir, config['data']['image_dir']),
        'test_labels': os.path.join(test_dir, config['data']['label_dir'])
    }
    
    missing_dirs = []
    for name, path in required_dirs.items():
        if not os.path.exists(path):
            missing_dirs.append(f"{name}: {path}")
    
    if missing_dirs:
        raise RuntimeError(
            "Missing required directories:\n" + 
            "\n".join(missing_dirs) +
            "\n\nPlease ensure your data is organized according to the README."
        )
    
    return required_dirs

def parse_wkt_polygon(wkt_str):
    """Parse WKT polygon string to numpy array of points."""
    coords_str = wkt_str.replace('POLYGON ((', '').replace('))', '')
    coords = [coord.split() for coord in coords_str.split(',')]
    return np.array(coords, dtype=np.float32)

class XBDDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, test_mode=False, image_size=(512, 512)):
        """
        Args:
            data_dir (str): Base directory containing the data
            split (str): 'train' or 'test'
            transform: Optional transforms to apply
            test_mode (bool): If True, only load a small subset for testing
            image_size (tuple): Target size for images and masks
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.test_mode = test_mode
        self.image_size = image_size
        self.valid_pairs = []
        self.skipped_pairs = []
        
        # Validate directories
        self.split_dir = os.path.join(data_dir, split)
        self.images_dir = os.path.join(self.split_dir, 'images')
        self.labels_dir = os.path.join(self.split_dir, 'labels')
        
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory not found at {self.images_dir}")
        if not os.path.exists(self.labels_dir):
            raise RuntimeError(f"Labels directory not found at {self.labels_dir}")
            
        # Get all pre-disaster images and validate pairs
        logger.info(f"Scanning for images in {self.images_dir}")
        all_pre_images = [f for f in os.listdir(self.images_dir) 
                         if f.endswith('_pre_disaster.png')]
        
        if len(all_pre_images) == 0:
            raise RuntimeError(f"No pre-disaster images found in {self.images_dir}")
            
        # Validate image/label pairs and post-disaster images
        for pre_img in all_pre_images:
            disaster_id = pre_img.replace('_pre_disaster.png', '')
            post_img = f"{disaster_id}_post_disaster.png"
            label_file = f"{disaster_id}_pre_disaster.json"
            
            pre_img_path = os.path.join(self.images_dir, pre_img)
            post_img_path = os.path.join(self.images_dir, post_img)
            label_path = os.path.join(self.labels_dir, label_file)
            
            skip_reason = None
            if not os.path.exists(post_img_path):
                skip_reason = f"Missing post-disaster image: {post_img}"
            elif not os.path.exists(label_path):
                skip_reason = f"Missing label file: {label_file}"
            else:
                try:
                    # Validate image can be opened
                    pre_image = Image.open(pre_img_path)
                    post_image = Image.open(post_img_path)
                    
                    # Validate label file format
                    with open(label_path) as f:
                        label_data = json.load(f)
                        if 'features' not in label_data or 'xy' not in label_data['features']:
                            skip_reason = f"Invalid label format in {label_file}"
                        else:
                            # Count buildings
                            buildings = [f for f in label_data['features']['xy'] 
                                       if f['properties']['feature_type'] == 'building']
                            if not buildings:
                                skip_reason = f"No building annotations in {label_file}"
                            
                except Exception as e:
                    skip_reason = f"Error processing {disaster_id}: {str(e)}"
            
            if skip_reason:
                self.skipped_pairs.append((disaster_id, skip_reason))
            else:
                self.valid_pairs.append({
                    'disaster_id': disaster_id,
                    'pre_img': pre_img,
                    'post_img': post_img,
                    'label_file': label_file
                })
        
        logger.info(f"Found {len(self.valid_pairs)} valid image/label pairs")
        if self.skipped_pairs:
            logger.warning(f"Skipped {len(self.skipped_pairs)} pairs:")
            for disaster_id, reason in self.skipped_pairs:
                logger.warning(f"  {disaster_id}: {reason}")
            
        if len(self.valid_pairs) == 0:
            raise RuntimeError("No valid image/label pairs found!")
            
        if test_mode:
            self.valid_pairs = self.valid_pairs[:4]
            logger.info("Test mode: using only 4 pairs")
            
        # Load one image to get dimensions
        sample_img = Image.open(os.path.join(self.images_dir, self.valid_pairs[0]['pre_img']))
        self.original_size = sample_img.size
            
    def __len__(self):
        return len(self.valid_pairs)
        
    def create_building_mask(self, label_data, scale_x, scale_y):
        """Create binary mask from building polygons with proper scaling."""
        mask = np.zeros(self.image_size, dtype=np.float32)
        building_count = 0
        
        try:
            for feature in label_data['features']['xy']:
                if feature['properties']['feature_type'] == 'building':
                    building_count += 1
                    # Parse and scale polygon coordinates
                    points = parse_wkt_polygon(feature['wkt'])
                    points[:, 0] *= scale_x
                    points[:, 1] *= scale_y
                    points = points.astype(np.int32)
                    
                    # Draw filled polygon
                    cv2.fillPoly(mask, [points], 1.0)
                    
            if building_count == 0:
                logger.warning("No buildings found in label data")
                
        except Exception as e:
            logger.error(f"Error creating mask: {str(e)}")
            raise
            
        return mask
        
    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        
        try:
            # Load pre and post disaster images
            pre_img_path = os.path.join(self.images_dir, pair['pre_img'])
            post_img_path = os.path.join(self.images_dir, pair['post_img'])
            
            pre_img = Image.open(pre_img_path).convert('RGB')
            post_img = Image.open(post_img_path).convert('RGB')
            
            # Calculate scaling factors
            scale_x = self.image_size[0] / self.original_size[0]
            scale_y = self.image_size[1] / self.original_size[1]
            
            # Load labels
            label_path = os.path.join(self.labels_dir, pair['label_file'])
            with open(label_path) as f:
                label_data = json.load(f)
                
            # Create building mask
            mask = self.create_building_mask(label_data, scale_x, scale_y)
            
            # Resize images to target size
            pre_img = pre_img.resize(self.image_size, Image.BILINEAR)
            post_img = post_img.resize(self.image_size, Image.BILINEAR)
            
            # Convert to tensors
            pre_img = transforms.ToTensor()(pre_img)
            post_img = transforms.ToTensor()(post_img)
            mask = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dimension
            
            if self.transform:
                pre_img = self.transform(pre_img)
                post_img = self.transform(post_img)
            
            return {
                'pre_image': pre_img,
                'post_image': post_img,
                'mask': mask,
                'disaster_id': pair['disaster_id']
            }
            
        except Exception as e:
            logger.error(f"Error loading data for {pair['disaster_id']}: {str(e)}")
            raise

class FilteredXBDDataset(XBDDataset):
    def __init__(self, data_dir, min_building_ratio=0.005, split='train', transform=None, test_mode=False, image_size=(512, 512)):
        """
        Args:
            data_dir (str): Base directory containing the data
            min_building_ratio (float): Minimum ratio of building pixels to total pixels (default: 0.5%)
            split (str): 'train' or 'test'
            transform: Optional transforms to apply
            test_mode (bool): If True, only load a small subset for testing
            image_size (tuple): Target size for images and masks
        """
        super().__init__(data_dir, split, transform, test_mode, image_size)
        self.min_building_ratio = min_building_ratio
        self.filtered_pairs = []
        self.filter_samples()
        
    def filter_samples(self):
        """Filter samples based on building ratio threshold."""
        logger.info(f"Filtering samples with building ratio threshold: {self.min_building_ratio:.1%}")
        original_count = len(self.valid_pairs)
        filtered_count = 0
        
        for pair in tqdm(self.valid_pairs, desc="Filtering samples"):
            try:
                # Load and process label file
                label_path = os.path.join(self.labels_dir, pair['label_file'])
                with open(label_path) as f:
                    label_data = json.load(f)
                
                # Calculate scaling factors
                scale_x = self.image_size[0] / self.original_size[0]
                scale_y = self.image_size[1] / self.original_size[1]
                
                # Create mask and calculate building ratio
                mask = self.create_building_mask(label_data, scale_x, scale_y)
                building_ratio = np.sum(mask > 0) / mask.size
                
                if building_ratio >= self.min_building_ratio:
                    self.filtered_pairs.append(pair)
                else:
                    filtered_count += 1
                    logger.debug(f"Filtered out {pair['disaster_id']} (building ratio: {building_ratio:.3%})")
                    
            except Exception as e:
                logger.warning(f"Error processing {pair['disaster_id']}: {str(e)}")
                filtered_count += 1
        
        # Replace valid_pairs with filtered_pairs
        self.valid_pairs = self.filtered_pairs
        
        logger.info(f"Filtered {filtered_count} samples with low building ratio")
        logger.info(f"Original dataset size: {original_count}")
        logger.info(f"Filtered dataset size: {len(self.valid_pairs)}")
        
        if len(self.valid_pairs) == 0:
            raise RuntimeError("No samples remain after filtering! Consider lowering the min_building_ratio threshold.")
            
    def __len__(self):
        return len(self.valid_pairs)

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

def get_device():
    if not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Using CPU.")
        return torch.device('cpu')
    print(f"[Info] CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    return torch.device('cuda')

def get_gpu_memory_usage():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / 1024**2  # MB

def get_system_stats():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    gpu_stats = GPUtil.getGPUs() if GPUtil.getGPUs() else []
    gpu_util = gpu_stats[0].load * 100 if gpu_stats else 0
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'gpu_util_percent': gpu_util,
        'gpu_memory_mb': get_gpu_memory_usage()
    }

def find_optimal_batch_size(dataset, model, max_batch_size=32, min_batch_size=1):
    """Find the largest batch size that fits in memory"""
    logger.info("Finding optimal batch size...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        logger.warning("No GPU detected, skipping batch size optimization")
        return min_batch_size
        
    batch_size = max_batch_size
    while batch_size >= min_batch_size:
        try:
            # Try loading a batch
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            batch = next(iter(loader))
            
            # Try a forward pass
            with torch.no_grad():
                model.to(device)
                pre_imgs = batch['pre_image'].to(device)
                post_imgs = batch['post_image'].to(device)
                _ = model(pre_imgs, post_imgs)
                
            logger.info(f"Found optimal batch size: {batch_size}")
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size //= 2
                torch.cuda.empty_cache()
            else:
                raise e
                
    logger.warning(f"Could not find working batch size, using minimum: {min_batch_size}")
    return min_batch_size

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with resource monitoring and error handling."""
    model.train()
    total_loss = 0
    processed_batches = 0
    failed_batches = 0
    
    # Initialize progress bar
    pbar = tqdm(loader, desc='Training', leave=True)
    
    # Track resource usage
    start_memory = get_system_stats()
    logger.info(f"Starting epoch - Memory usage: {start_memory['memory_percent']}%")
    if torch.cuda.is_available():
        logger.info(f"GPU memory: {get_gpu_memory_usage()}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Move data to device
            pre_images = batch['pre_image'].to(device)
            post_images = batch['post_image'].to(device)
            masks = batch['mask'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with automatic mixed precision if enabled
            context = autocast() if scaler else nullcontext()
            with context:
                outputs = model(pre_images, post_images)  # Raw logits
                loss = criterion(outputs, masks)  # BCEWithLogitsLoss handles the sigmoid
            
            # Check for NaN loss
            if torch.isnan(loss):
                logger.error(f"NaN loss detected in batch {batch_idx}")
                failed_batches += 1
                continue
                
            # Backward pass with gradient scaling if enabled
            if scaler:
                scaler.scale(loss).backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            processed_batches += 1
            
            # Calculate and display current metrics
            avg_loss = total_loss / processed_batches
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'failed': failed_batches
            })
            
            # Log resource usage every 10 batches
            if batch_idx % 10 == 0:
                current_memory = get_system_stats()
                logger.info(f"Batch {batch_idx} - Memory: {current_memory['memory_percent']}%")
                if torch.cuda.is_available():
                    logger.info(f"GPU memory: {get_gpu_memory_usage()}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.error(f"GPU OOM in batch {batch_idx}. Skipping batch.")
                failed_batches += 1
                continue
            else:
                logger.error(f"Runtime error in batch {batch_idx}: {str(e)}")
                failed_batches += 1
                continue
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            failed_batches += 1
            continue
    
    # Final stats
    avg_loss = total_loss / processed_batches if processed_batches > 0 else float('inf')
    end_memory = get_system_stats()
    
    stats = {
        'average_loss': avg_loss,
        'processed_batches': processed_batches,
        'failed_batches': failed_batches,
        'memory_usage': end_memory['memory_percent'],
        'gpu_memory': get_gpu_memory_usage() if torch.cuda.is_available() else 'N/A'
    }
    
    logger.info("Epoch completed:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return stats

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train building localization model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--min_building_ratio', type=float, default=0.005, help='Minimum building ratio threshold')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with limited samples')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset with filtering
    dataset = FilteredXBDDataset(
        data_dir=args.data_dir,
        min_building_ratio=args.min_building_ratio,
        test_mode=args.test_mode
    )
    
    # Set device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Initialize loader
    try:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Number of batches: {len(loader)}")
        
    except Exception as e:
        logger.error(f"Failed to initialize dataset: {str(e)}")
        return
    
    # Initialize model
    try:
        model = SimpleUNet().to(device)
        criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        
        # Initialize mixed precision if available
        scaler = GradScaler() if device.type == 'cuda' else None
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return
    
    # Training loop
    try:
        for epoch in range(args.epochs):
            logger.info(f"\nStarting epoch {epoch+1}/{args.epochs}")
            
            stats = train_epoch(
                model=model,
                loader=loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler
            )
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stats': stats
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
    finally:
        # Save final model
        try:
            final_path = os.path.join(args.output_dir, 'final_model.pth')
            torch.save(model.state_dict(), final_path)
            logger.info(f"Saved final model to {final_path}")
        except Exception as e:
            logger.error(f"Failed to save final model: {str(e)}")

if __name__ == '__main__':
    main() 