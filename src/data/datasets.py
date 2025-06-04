"""Dataset classes for xBD pipeline."""

import os
import json
import torch
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from torch.utils.data import Dataset
import albumentations as A
from shapely.geometry import Polygon, mapping
import cv2

class DamageDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 transform: Optional[A.Compose] = None,
                 phase: str = 'train',
                 disaster_name: Optional[str] = None):
        """
        xBD dataset for damage classification.
        
        Args:
            data_dir: Path to xBD dataset root
            transform: Albumentations transforms
            phase: 'train' or 'val'
            disaster_name: Optional disaster name to filter data
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.phase = phase
        
        # Get all disaster directories
        if disaster_name:
            self.disaster_dirs = [self.data_dir / disaster_name]
        else:
            self.disaster_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        # Collect all image pairs and their labels
        self.samples = []
        for disaster_dir in self.disaster_dirs:
            images_dir = disaster_dir / 'images'
            labels_dir = disaster_dir / 'labels'
            
            if not (images_dir.exists() and labels_dir.exists()):
                continue
            
            # Get all pre-post image pairs
            for pre_img_path in images_dir.glob('*_pre_disaster.png'):
                post_img_path = images_dir / pre_img_path.name.replace('_pre_disaster', '_post_disaster')
                label_path = labels_dir / f"{pre_img_path.stem.replace('_pre_disaster', '')}.json"
                
                if post_img_path.exists() and label_path.exists():
                    self.samples.append({
                        'pre_image_path': pre_img_path,
                        'post_image_path': post_img_path,
                        'label_path': label_path,
                        'disaster': disaster_dir.name
                    })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load images
        with rasterio.open(sample['pre_image_path']) as src:
            pre_image = src.read().transpose(1, 2, 0)  # (H, W, C)
        with rasterio.open(sample['post_image_path']) as src:
            post_image = src.read().transpose(1, 2, 0)  # (H, W, C)
        
        # Load damage labels
        with open(sample['label_path']) as f:
            data = json.load(f)
            
        # Create damage mask
        damage_mask = np.zeros(pre_image.shape[:2], dtype=np.uint8)
        for feature in data['features']:
            if feature['properties']['subtype'] == 'building':
                damage_level = feature['properties'].get('damage_level', 0)
                polygon = Polygon(feature['geometry']['coordinates'][0])
                coords = np.array(polygon.exterior.coords, dtype=np.int32)
                cv2.fillPoly(damage_mask, [coords], damage_level + 1)  # Add 1 to avoid 0 confusion
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=post_image, mask=damage_mask)
            post_image = transformed['image']
            damage_mask = transformed['mask']
        
        return {
            'pre_image': torch.from_numpy(pre_image).permute(2, 0, 1),  # (C, H, W)
            'post_image': torch.from_numpy(post_image).permute(2, 0, 1),  # (C, H, W)
            'damage_mask': torch.from_numpy(damage_mask - 1).long(),  # Convert back to 0-based
            'disaster': sample['disaster'],
            'image_path': str(sample['post_image_path']),
            'mask_path': str(sample['label_path'])
        }
    
    def get_damage_levels(self, idx: int) -> List[int]:
        """Get damage levels for all buildings in a sample."""
        sample = self.samples[idx]
        with open(sample['label_path']) as f:
            data = json.load(f)
        
        damage_levels = []
        for feature in data['features']:
            if feature['properties']['subtype'] == 'building':
                damage_level = feature['properties'].get('damage_level', 0)
                damage_levels.append(damage_level)
        
        return damage_levels 