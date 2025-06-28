import os
import torch
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
from shapely.wkt import loads
from shapely.geometry import Polygon
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
                self.samples.append({
                    'image': os.path.join(self.image_dir, img_file),
                    'label': label_path
                })
        
        print(f'{self.split} samples: {len(self.samples)}')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = cv2.imread(sample['image'])
        if image is None:
            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[:2] != (1024, 1024):
                image = cv2.resize(image, (1024, 1024))
        
        mask = self._create_mask(sample['label'])
        
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        mask = mask.astype(np.float32)
        mask = np.expand_dims(mask, axis=0)
        
        return torch.from_numpy(image), torch.from_numpy(mask)
    
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

def get_proper_data_loaders(data_dir, batch_size=2):
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
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=3
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f'Error creating data loaders: {e}')
        return None, None 