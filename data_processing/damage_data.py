import os
import torch
import numpy as np
import json
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from shapely.wkt import loads
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings('ignore')

class DamageDataset(Dataset):
    def __init__(self, data_dir, split='train', augment=True, patch_size=64):
        self.data_dir = data_dir
        self.split = split
        self.augment = augment
        self.patch_size = patch_size
        self.image_dir = os.path.join(data_dir, split, 'images')
        self.label_dir = os.path.join(data_dir, split, 'labels')
        
        self.damage_classes = {
            'no-damage': 0,
            'minor-damage': 1, 
            'major-damage': 2,
            'destroyed': 3
        }
        
        self.patches = []
        self.labels = []
        self._extract_building_patches()
        self._balance_classes()
        
    def _extract_building_patches(self):
        if not os.path.exists(self.image_dir) or not os.path.exists(self.label_dir):
            print(f'Missing directories: {self.image_dir} or {self.label_dir}')
            return
            
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        
        for img_file in image_files:
            if 'post_disaster' not in img_file:
                continue
                
            base_name = img_file.replace('.png', '')
            label_file = base_name + '.json'
            label_path = os.path.join(self.label_dir, label_file)
            
            if os.path.exists(label_path):
                self._process_image(os.path.join(self.image_dir, img_file), label_path)
        
        print(f'{self.split} patches extracted: {len(self.patches)}')
        self._print_class_distribution()
    
    def _process_image(self, image_path, label_path):
        image = cv2.imread(image_path)
        if image is None:
            return
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (1024, 1024):
            image = cv2.resize(image, (1024, 1024))
        
        try:
            with open(label_path, 'r') as f:
                data = json.load(f)
            
            if 'features' in data and 'xy' in data['features']:
                features = data['features']['xy']
                
                for feature in features:
                    if 'wkt' in feature and 'properties' in feature:
                        props = feature['properties']
                        if props.get('feature_type') == 'building':
                            damage_type = props.get('subtype', 'no-damage')
                            
                            if damage_type in self.damage_classes:
                                patch = self._extract_patch(image, feature['wkt'])
                                if patch is not None:
                                    self.patches.append(patch)
                                    self.labels.append(self.damage_classes[damage_type])
        except:
            pass
    
    def _extract_patch(self, image, wkt_str):
        try:
            geom = loads(wkt_str)
            if isinstance(geom, Polygon) and geom.is_valid:
                coords = np.array(geom.exterior.coords, dtype=np.int32)
                
                x_min, y_min = coords.min(axis=0)
                x_max, y_max = coords.max(axis=0)
                
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                
                half_size = self.patch_size // 2
                
                x1 = max(0, center_x - half_size)
                y1 = max(0, center_y - half_size) 
                x2 = min(image.shape[1], center_x + half_size)
                y2 = min(image.shape[0], center_y + half_size)
                
                patch = image[y1:y2, x1:x2]
                
                if patch.shape[0] > 0 and patch.shape[1] > 0:
                    patch = cv2.resize(patch, (self.patch_size, self.patch_size))
                    return patch
        except:
            pass
        return None
    
    def _balance_classes(self):
        if len(self.patches) == 0:
            return
            
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        max_samples_per_class = 10000
        
        balanced_patches = []
        balanced_labels = []
        
        for class_id in range(4):
            class_patches = [p for i, p in enumerate(self.patches) if self.labels[i] == class_id]
            
            if len(class_patches) > 0:
                if len(class_patches) > max_samples_per_class:
                    indices = np.random.choice(len(class_patches), max_samples_per_class, replace=False)
                    class_patches = [class_patches[i] for i in indices]
                
                min_samples = min(len(class_patches), 2000)
                while len(class_patches) < min_samples:
                    idx = np.random.randint(len(class_patches))
                    augmented_patch = self._augment_patch(class_patches[idx])
                    class_patches.append(augmented_patch)
                
                class_labels = [class_id] * len(class_patches)
                balanced_patches.extend(class_patches)
                balanced_labels.extend(class_labels)
        
        self.patches = balanced_patches
        self.labels = balanced_labels
        
        print(f'Balanced dataset: {len(self.patches)} patches')
        self._print_class_distribution()
    
    def _augment_patch(self, patch):
        h, w = patch.shape[:2]
        
        if np.random.random() > 0.5:
            patch = cv2.flip(patch, 1)
        
        if np.random.random() > 0.5:
            patch = cv2.flip(patch, 0)
        
        if np.random.random() > 0.5:
            angle = np.random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            patch = cv2.warpAffine(patch, M, (w, h))
        
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            patch = np.clip(patch * brightness, 0, 255).astype(np.uint8)
        
        return patch
    
    def _print_class_distribution(self):
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
        for i, name in enumerate(class_names):
            count = class_counts.get(i, 0)
            print(f'{name}: {count}')
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx].copy()
        label = self.labels[idx]
        
        if self.augment and self.split == 'train':
            patch = self._augment_patch(patch)
        
        patch = patch.astype(np.float32) / 255.0
        patch = np.transpose(patch, (2, 0, 1))
        
        return torch.from_numpy(patch), torch.tensor(label, dtype=torch.long)

def get_damage_data_loaders(data_dir, batch_size=32, patch_size=64, num_workers=4, prefetch_factor=2):
    try:
        train_dataset = DamageDataset(data_dir, 'train', augment=True, patch_size=patch_size)
        val_dataset = DamageDataset(data_dir, 'test', augment=False, patch_size=patch_size)
        
        if len(train_dataset) == 0:
            print('No training patches found!')
            return None, None
        
        # Fixed: Handle prefetch_factor correctly
        dataloader_kwargs = {
            'batch_size': batch_size,
            'pin_memory': True,
            'drop_last': True
        }
        
        if num_workers > 0:
            dataloader_kwargs.update({
                'num_workers': num_workers,
                'prefetch_factor': prefetch_factor,
                'persistent_workers': True
            })
        else:
            dataloader_kwargs['num_workers'] = 0
        
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **dataloader_kwargs
        )
        
        val_dataloader_kwargs = dataloader_kwargs.copy()
        val_dataloader_kwargs.update({
            'shuffle': False,
            'drop_last': False
        })
        
        val_loader = DataLoader(
            val_dataset,
            **val_dataloader_kwargs
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f'Error creating damage data loaders: {e}')
        return None, None

 