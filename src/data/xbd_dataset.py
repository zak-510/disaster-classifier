import os
import json
import torch
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Tuple, Optional
from torch.utils.data import Dataset
import albumentations as A
from shapely.geometry import Polygon, mapping
import cv2

class XBDDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 transform: Optional[A.Compose] = None,
                 phase: str = 'train',
                 disaster_name: Optional[str] = None):
        """
        xBD dataset for building localization.
        
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
        
        # Collect all pre-disaster images and their labels
        self.samples = []
        for disaster_dir in self.disaster_dirs:
            images_dir = disaster_dir / 'images'
            labels_dir = disaster_dir / 'labels'
            
            if not (images_dir.exists() and labels_dir.exists()):
                continue
            
            # Get all pre-disaster images
            for img_path in images_dir.glob('*_pre_disaster.png'):
                label_path = labels_dir / f"{img_path.stem.replace('_pre_disaster', '')}.json"
                if label_path.exists():
                    self.samples.append({
                        'image_path': img_path,
                        'label_path': label_path,
                        'disaster': disaster_dir.name
                    })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        with rasterio.open(sample['image_path']) as src:
            image = src.read().transpose(1, 2, 0)  # (H, W, C)
        
        # Load and convert polygons to mask
        mask = self._load_mask(sample['label_path'], image.shape[:2])
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': image,
            'mask': mask.unsqueeze(0),  # Add channel dimension
            'disaster': sample['disaster'],
            'image_path': str(sample['image_path'])
        }
    
    def _load_mask(self, label_path: Path, image_shape: Tuple[int, int]) -> np.ndarray:
        """Convert GeoJSON polygons to binary mask."""
        with open(label_path) as f:
            data = json.load(f)
        
        # Create empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Draw each building polygon
        for feature in data['features']:
            if feature['properties']['subtype'] == 'building':
                polygon = Polygon(feature['geometry']['coordinates'][0])
                # Convert polygon to pixel coordinates
                coords = np.array(polygon.exterior.coords, dtype=np.int32)
                cv2.fillPoly(mask, [coords], 1)
        
        return mask
    
    def get_polygons(self, idx: int) -> Dict:
        """Get original polygon annotations for a sample."""
        sample = self.samples[idx]
        with open(sample['label_path']) as f:
            return json.load(f)
    
    @staticmethod
    def mask_to_polygons(mask: np.ndarray, min_area: float = 100) -> Dict:
        """Convert binary mask to GeoJSON polygons."""
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to polygons
        features = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            
            # Simplify polygon
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Create GeoJSON feature
            polygon = Polygon(approx.reshape(-1, 2))
            feature = {
                'type': 'Feature',
                'geometry': mapping(polygon),
                'properties': {
                    'subtype': 'building',
                    'confidence': 1.0
                }
            }
            features.append(feature)
        
        return {
            'type': 'FeatureCollection',
            'features': features
        } 