import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Tuple

def get_localization_augmentations(config: Dict[str, Any], phase: str = 'train') -> A.Compose:
    """
    Get augmentation pipeline for localization model.
    
    Args:
        config: Augmentation configuration from model config
        phase: 'train' or 'val'
    
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    if phase == 'train':
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=config['augmentation']['rotate_limit'],
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=config['augmentation']['brightness_limit'],
                    contrast_limit=config['augmentation']['contrast_limit'],
                    p=0.5
                ),
                A.HueSaturationValue(p=0.5),
            ], p=0.3),
            A.Resize(config['image_size'][0], config['image_size'][1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(config['image_size'][0], config['image_size'][1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def get_classification_augmentations(config: Dict[str, Any], phase: str = 'train') -> A.Compose:
    """
    Get augmentation pipeline for damage classification model.
    
    Args:
        config: Augmentation configuration from model config
        phase: 'train' or 'val'
    
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    if phase == 'train':
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=config['augmentation']['rotate_limit'],
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=config['augmentation']['brightness_limit'],
                    contrast_limit=config['augmentation']['contrast_limit'],
                    p=0.5
                ),
                A.HueSaturationValue(p=0.5),
            ], p=0.3),
            A.RandomResizedCrop(
                height=config['image_size'][0],
                width=config['image_size'][1],
                scale=(0.8, 1.0),
                p=0.5
            ),
            A.Resize(config['image_size'][0], config['image_size'][1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(config['image_size'][0], config['image_size'][1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]) 