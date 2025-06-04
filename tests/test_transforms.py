import pytest
import numpy as np
import torch
from src.utils.transforms import (
    normalize_image,
    augment_image_mask,
    resize_image_mask,
    convert_to_tensor
)

def test_normalize_image(sample_image):
    normalized = normalize_image(sample_image)
    assert normalized.dtype == np.float32
    assert normalized.max() <= 1.0
    assert normalized.min() >= 0.0

def test_augment_image_mask(sample_image, sample_mask):
    augmented_img, augmented_mask = augment_image_mask(
        sample_image, 
        sample_mask,
        do_flip=True,
        do_rotate=True
    )
    assert augmented_img.shape == sample_image.shape
    assert augmented_mask.shape == sample_mask.shape
    assert augmented_mask.dtype == sample_mask.dtype

def test_resize_image_mask(sample_image, sample_mask):
    target_size = (256, 256)
    resized_img, resized_mask = resize_image_mask(
        sample_image,
        sample_mask,
        target_size=target_size
    )
    assert resized_img.shape[:2] == target_size
    assert resized_mask.shape[:2] == target_size

def test_convert_to_tensor(sample_image, sample_mask):
    img_tensor, mask_tensor = convert_to_tensor(sample_image, sample_mask)
    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(mask_tensor, torch.Tensor)
    assert img_tensor.shape[0] == 3  # Channels first
    assert mask_tensor.shape[0] == 1  # Single channel

def test_invalid_inputs():
    with pytest.raises(ValueError):
        normalize_image(np.ones((10, 10, 4)))  # Invalid channels
    
    with pytest.raises(ValueError):
        resize_image_mask(
            np.ones((100, 100, 3)),
            np.ones((50, 50)),  # Mismatched sizes
            (256, 256)
        )

@pytest.mark.parametrize("target_size", [
    (224, 224),
    (512, 512),
    (1024, 1024)
])
def test_resize_different_sizes(sample_image, sample_mask, target_size):
    resized_img, resized_mask = resize_image_mask(
        sample_image,
        sample_mask,
        target_size=target_size
    )
    assert resized_img.shape[:2] == target_size
    assert resized_mask.shape[:2] == target_size 