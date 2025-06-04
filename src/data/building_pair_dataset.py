import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple
import albumentations as A

class BuildingPairDataset(Dataset):
    def __init__(self, 
                 pre_images: List[np.ndarray],
                 post_images: List[np.ndarray],
                 bboxes: List[Tuple[int, int, int, int]],
                 transform: A.Compose,
                 padding: int = 10):
        """
        Dataset for pre-post disaster building image pairs.
        
        Args:
            pre_images: List of pre-disaster images
            post_images: List of post-disaster images
            bboxes: List of building bounding boxes (x1, y1, x2, y2)
            transform: Albumentations transforms
            padding: Padding around building crops
        """
        self.pre_images = pre_images
        self.post_images = post_images
        self.bboxes = bboxes
        self.transform = transform
        self.padding = padding
        
    def __len__(self) -> int:
        return len(self.bboxes)
    
    def crop_building(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop building from image with padding."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(w, x2 + self.padding)
        y2 = min(h, y2 + self.padding)
        
        return image[y1:y2, x1:x2]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get image pair and bbox
        pre_image = self.pre_images[idx]
        post_image = self.post_images[idx]
        bbox = self.bboxes[idx]
        
        # Crop buildings
        pre_crop = self.crop_building(pre_image, bbox)
        post_crop = self.crop_building(post_image, bbox)
        
        # Apply transforms
        pre_crop = self.transform(image=pre_crop)['image']
        post_crop = self.transform(image=post_crop)['image']
        
        # Stack pre and post crops
        pair = np.stack([pre_crop, post_crop])
        
        return {
            'image_pair': torch.from_numpy(pair),
            'idx': idx
        } 