import os
import torch
import pytest
import numpy as np
from pathlib import Path
import json
import shutil

from src.models.localization.unet import UNet, Up
from src.run_inference import process_image, run_inference, generate_statistics
from src.models.damage_classification import DamageClassifier

# Set deterministic behavior
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def test_data_dir(tmp_path):
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir

@pytest.fixture
def test_output_dir(tmp_path):
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture
def mock_image(test_data_dir):
    # Create a simple test image
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = 255  # White square in middle
    img_path = test_data_dir / "test_image.png"
    import cv2
    cv2.imwrite(str(img_path), img)
    return img_path

def test_unet_channel_fix(device):
    """Test that the UNet Up module handles channels correctly."""
    # Test both bilinear and non-bilinear modes
    for bilinear in [True, False]:
        features = [64, 128, 256, 512, 1024]
        up = Up(features[4], features[3], bilinear=bilinear).to(device)
        
        # Create test tensors
        x1 = torch.randn(1, features[4], 4, 4).to(device)  # Higher level features
        x2 = torch.randn(1, features[3], 8, 8).to(device)  # Skip connection features
        
        # This should not raise any RuntimeError
        output = up(x1, x2)
        assert output.shape == (1, features[3], 8, 8), f"Wrong output shape: {output.shape}"

def test_process_image_debug_dir(mock_image, test_output_dir, device):
    """Test that process_image handles debug_dir correctly."""
    # Initialize test models
    loc_model = UNet(in_channels=3, out_channels=1).to(device)
    damage_model = DamageClassifier(num_classes=4).to(device)
    
    # Test with debug_dir=None
    result = process_image(
        Path(mock_image),
        loc_model,
        damage_model,
        device,
        debug_dir=None,
        threshold=0.5
    )
    assert 'processed_mask' in result
    assert not (test_output_dir / 'debug').exists()
    
    # Test with debug_dir provided
    debug_dir = test_output_dir / 'debug'
    debug_dir.mkdir()
    result = process_image(
        Path(mock_image),
        loc_model,
        damage_model,
        device,
        debug_dir=debug_dir,
        threshold=0.5
    )
    assert (debug_dir / f"raw_mask_{Path(mock_image).stem}.png").exists()

def test_generate_statistics(test_output_dir):
    """Test the generate_statistics function."""
    test_results = [
        {"damage_level": 1, "confidence": 0.8},
        {"damage_level": 2, "confidence": 0.9},
        {"damage_level": 1, "confidence": 0.7}
    ]
    
    stats_file = test_output_dir / "stats.json"
    generate_statistics(test_results, stats_file)
    
    with open(stats_file) as f:
        stats = json.load(f)
    
    assert stats["total"] == 3
    assert stats["damage_levels"] == {"1": 2, "2": 1}
    assert abs(stats["avg_confidence"] - 0.8) < 1e-6

def test_run_inference_pipeline(test_data_dir, test_output_dir, mock_image, device):
    """Test the complete inference pipeline."""
    # Create mock model checkpoints
    loc_model = UNet(in_channels=3, out_channels=1)
    damage_model = DamageClassifier(num_classes=4)
    
    loc_checkpoint = test_data_dir / "loc_model.pth"
    damage_checkpoint = test_data_dir / "damage_model.pth"
    
    torch.save({"model_state_dict": loc_model.state_dict()}, loc_checkpoint)
    torch.save({"model_state_dict": damage_model.state_dict()}, damage_checkpoint)
    
    # Set test mode
    os.environ['TESTING'] = '1'
    
    # Run inference
    run_inference(
        test_data_dir,
        test_output_dir,
        loc_checkpoint,
        damage_checkpoint,
        threshold=0.5,
        batch_size=1,
        debug=True
    )
    
    # Check outputs
    assert (test_output_dir / 'localization' / 'localization_results.npy').exists()
    assert (test_output_dir / 'classification' / 'classification_results.json').exists()
    assert (test_output_dir / 'classification' / 'statistics.json').exists()
    
    # Clean up
    os.environ.pop('TESTING') 