import os
import sys
import yaml
import json
import torch
import numpy as np
import pytest
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock
import psutil
import tempfile
import shutil

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import validate_inputs
from src.run_pipeline import run_pipeline
from src.run_inference import run_inference, process_image, load_model
from src.utils.report_generator import HTMLReportGenerator
from src.models.unet import UNet
from src.models.classifier import DamageClassifier

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        'data': {
            'input_dir': 'data/xBD',
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'image_size': [1024, 1024],
            'batch_size': 8,
            'num_workers': 4
        },
        'models': {
            'localization': {
                'checkpoint': 'models/localization/checkpoint.pth',
                'output_name': 'localization_model.pth',
                'architecture': 'resnet50',
                'pretrained': True,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'num_epochs': 100,
                'early_stopping_patience': 10
            },
            'damage': {
                'checkpoint': 'models/damage/checkpoint.pth',
                'output_name': 'damage_model.pth',
                'architecture': 'resnet50',
                'pretrained': True,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'num_epochs': 100,
                'early_stopping_patience': 10,
                'damage_levels': {
                    '0': {'name': 'no_damage', 'min_confidence': 0.7},
                    '1': {'name': 'minor_damage', 'min_confidence': 0.7},
                    '2': {'name': 'major_damage', 'min_confidence': 0.7},
                    '3': {'name': 'destroyed', 'min_confidence': 0.7}
                }
            }
        },
        'inference': {
            'threshold': 0.5,
            'batch_size': 16,
            'device': 'cuda',
            'save_debug': True,
            'debug_dir': 'debug',
            'output_format': 'geojson',
            'min_confidence': 0.7
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(levelname)s - %(message)s',
            'file': 'pipeline.log'
        }
    }

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

def create_sample_data(temp_dir: Path):
    """Create sample data for testing."""
    # Create directory structure
    data_dir = temp_dir / 'data' / 'xBD'
    data_dir.mkdir(parents=True)
    
    # Create sample images
    img_dir = data_dir / 'images'
    img_dir.mkdir()
    
    # Create a sample image
    img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / 'sample.png'), img)
    
    return data_dir

def test_config_validation(sample_config, temp_dir):
    """Test configuration validation."""
    # Test valid config
    assert validate_inputs(sample_config, test_mode=True) is True
    
    # Test missing required paths
    invalid_config = sample_config.copy()
    invalid_config['data']['input_dir'] = 'nonexistent/path'
    assert validate_inputs(invalid_config, test_mode=True) is True  # Should pass in test mode
    
    # Test missing required section
    invalid_config = sample_config.copy()
    del invalid_config['data']
    assert validate_inputs(invalid_config, test_mode=True) is False  # Should fail even in test mode

def test_path_handling(temp_dir):
    """Test path handling across different OS environments."""
    # Create test paths
    test_path = temp_dir / 'test' / 'path' / 'file.txt'
    test_path.parent.mkdir(parents=True)
    test_path.write_text('test')
    
    # Test path operations
    assert test_path.exists()
    assert test_path.is_file()
    assert test_path.parent.is_dir()
    
    # Test path joining
    new_path = test_path.parent / 'new_file.txt'
    assert str(new_path).replace('\\', '/') == str(test_path.parent / 'new_file.txt').replace('\\', '/')

def test_error_handling(temp_dir):
    """Test error handling with invalid inputs."""
    # Test invalid model path
    with pytest.raises(FileNotFoundError):
        load_model(temp_dir / 'nonexistent.pth', 'localization', torch.device('cpu'))
    
    # Test invalid image
    with pytest.raises(ValueError):
        process_image(
            temp_dir / 'nonexistent.png',
            MagicMock(),
            MagicMock(),
            torch.device('cpu'),
            temp_dir,
            threshold=0.1
        )

def test_memory_usage(temp_dir):
    """Test memory usage during processing."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Create and process a large image
    img = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    img_path = temp_dir / 'large_image.png'
    cv2.imwrite(str(img_path), img)
    
    # Process image
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(1, 1, 2048, 2048)
    
    process_image(
        img_path,
        mock_model,
        mock_model,
        torch.device('cpu'),
        temp_dir,
        threshold=0.1
    )
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Assert memory usage is within reasonable bounds (e.g., < 1GB)
    assert memory_increase < 1024 * 1024 * 1024

def test_output_validation(temp_dir):
    """Test output validation."""
    # Create sample data
    data_dir = create_sample_data(temp_dir)

    # Run inference
    output_dir = temp_dir / 'output'
    output_dir.mkdir()

    # Create mock model files using real model state dicts
    localization_model_path = temp_dir / 'localization_model.pth'
    damage_model_path = temp_dir / 'damage_model.pth'
    
    unet = UNet(in_channels=3, out_channels=1)
    torch.save({'model_state_dict': unet.state_dict()}, localization_model_path)
    classifier = DamageClassifier(num_classes=4)
    torch.save({'model_state_dict': classifier.state_dict()}, damage_model_path)

    # Mock models
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(1, 1, 1024, 1024)

    # Run inference
    run_inference(
        data_dir / 'images',
        output_dir,
        localization_model_path,
        damage_model_path,
        threshold=0.1,
        batch_size=4,
        debug=True
    )

    # Verify outputs
    assert (output_dir / 'localization' / 'masks').exists()
    assert (output_dir / 'classification' / 'results.json').exists()
    assert (output_dir / 'visualizations').exists()

def test_report_generation(temp_dir):
    """Test HTML report generation."""
    # Create sample statistics
    stats = {
        'total_buildings': 100,
        'damage_levels': {
            '0': {'count': 50, 'avg_confidence': 0.85},
            '1': {'count': 30, 'avg_confidence': 0.75},
            '2': {'count': 15, 'avg_confidence': 0.65},
            '3': {'count': 5, 'avg_confidence': 0.55}
        },
        'min_confidence': 0.7
    }

    # Create sample images
    vis_dir = temp_dir / 'visualizations'
    vis_dir.mkdir()
    for i in range(3):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(str(vis_dir / f'sample_{i}.png'), img)

    # Generate report
    report_generator = HTMLReportGenerator(
        output_dir=temp_dir,
        disaster_name='sample_disaster'
    )
    
    # Generate report with sample images
    example_images = [str(vis_dir / f'sample_{i}.png') for i in range(3)]
    report_path = report_generator.generate_report(stats, example_images)
    
    # Verify report was generated
    assert report_path.exists()
    assert report_path.suffix == '.html'

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 