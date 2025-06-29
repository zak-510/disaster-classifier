import os
import torch
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
from shapely.wkt import loads
from shapely.geometry import Polygon
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

class XBDDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.image_dir = os.path.join(data_dir, split, 'images')
        self.label_dir = os.path.join(data_dir, split, 'labels')
        
        self.samples = []
        self._load_samples()
        
        # Define augmentations
        if split == 'train':
            self.transform = A.Compose([
                # Spatial augmentations
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.2,  # Increased scale variation
                    rotate_limit=45,
                    p=0.7  # Increased probability
                ),
                # Reduce aggressive deformations
                A.OneOf([
                    A.ElasticTransform(
                        alpha=60,  # Reduced deformation
                        sigma=60 * 0.05,
                        alpha_affine=60 * 0.03,
                        p=0.3
                    ),
                    A.GridDistortion(p=0.3),
                    A.OpticalDistortion(
                        distort_limit=0.5,  # Reduced distortion
                        shift_limit=0.2,
                        p=0.3
                    ),
                ], p=0.2),
                # Reduce color augmentations
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.5),
                ], p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0
                )
            ], p=1.0)
        else:
            self.transform = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0
                )
            ])
    
    def _load_samples(self):
        if not os.path.exists(self.image_dir) or not os.path.exists(self.label_dir):
            print(f'Missing directories: {self.image_dir} or {self.label_dir}')
            return
            
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        
        for img_file in image_files:
            base_name = img_file.replace('.png', '')
            label_file = base_name + '.json'
            label_path = os.path.join(self.label_dir, label_file)
            
            if os.path.exists(label_path):
                # Check if the label contains any buildings
                try:
                    with open(label_path, 'r') as f:
                        data = json.load(f)
                    if 'features' in data and 'xy' in data['features']:
                        features = data['features']['xy']
                        if len(features) > 0:  # Only add if there are buildings
                            self.samples.append({
                                'image': os.path.join(self.image_dir, img_file),
                                'label': label_path
                            })
                except:
                    continue
        
        if self.split == 'train':
            # Duplicate samples with buildings to increase their frequency
            building_samples = [s for s in self.samples if self._has_buildings(s['label'])]
            self.samples.extend(building_samples)  # Add building samples twice
        
        print(f'{self.split} samples: {len(self.samples)}')
    
    def _has_buildings(self, label_path):
        try:
            with open(label_path, 'r') as f:
                data = json.load(f)
            if 'features' in data and 'xy' in data['features']:
                return len(data['features']['xy']) > 0
        except:
            pass
        return False
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image'])
        if image is None:
            raise ValueError(f"Could not load image: {sample['image']}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (1024, 1024):
            image = cv2.resize(image, (1024, 1024))
        
        # Create mask
        mask = self._create_mask(sample['label'])
        
        # Skip samples with empty masks
        if mask.sum() == 0:
            # Get a different sample
            return self.__getitem__((idx + 1) % len(self))
        
        # Apply augmentations
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Convert to tensor format
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask
    
    def _create_mask(self, label_path):
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        
        try:
            with open(label_path, 'r') as f:
                data = json.load(f)
            
            if 'features' in data and 'xy' in data['features']:
                features = data['features']['xy']
                
                for feature in features:
                    if 'wkt' in feature:
                        try:
                            geom = loads(feature['wkt'])
                            if isinstance(geom, Polygon) and geom.is_valid:
                                coords = np.array(geom.exterior.coords, dtype=np.int32)
                                cv2.fillPoly(mask, [coords], 1)
                        except:
                            continue
        except:
            pass
        
        return mask

def get_proper_data_loaders(data_dir, batch_size=8, num_workers=6, prefetch_factor=3):
    try:
        train_dataset = XBDDataset(data_dir, 'train')
        val_dataset = XBDDataset(data_dir, 'test')
        
        if len(train_dataset) == 0:
            print('No training samples found!')
            return None, None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f'Error creating data loaders: {e}')
        return None, None 