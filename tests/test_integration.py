"""Integration tests for xBD pipeline."""

import os
import sys
import pytest
from pathlib import Path
import shutil
import tempfile
import numpy as np
import cv2
import json
import yaml
import requests
from flask import Flask
import threading
import time
import torch
from datetime import datetime

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.cli.main import cli
from src.dashboard.app import app as dashboard_app
from src.models.unet import UNet
from src.models.classifier import DamageClassifier

@pytest.fixture
def app():
    """Create a Flask app for testing."""
    dashboard_app.config["TESTING"] = True
    return dashboard_app

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_data(temp_dir):
    """Create sample data for testing."""
    # Create directory structure
    data_dir = temp_dir / 'data'
    data_dir.mkdir()
    
    # Create sample images
    img_dir = data_dir / 'images'
    img_dir.mkdir()
    
    # Create a sample image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / 'sample.png'), img)
    
    return data_dir

@pytest.fixture
def sample_config(temp_dir):
    """Create a sample configuration file."""
    config = {
        'data': {
            'input_dir': str(temp_dir / 'data'),
            'image_size': [256, 256],
            'batch_size': 2
        },
        'models': {
            'localization': {
                'architecture': 'unet',
                'pretrained': True,
                'checkpoint': str(temp_dir / 'models' / 'localization' / 'checkpoint.pth')
            },
            'damage': {
                'architecture': 'resnet50',
                'pretrained': True,
                'checkpoint': str(temp_dir / 'models' / 'damage' / 'checkpoint.pth'),
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
            'batch_size': 2,
            'device': 'cpu',
            'save_debug': True,
            'min_confidence': 0.7
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(levelname)s - %(message)s',
            'file': 'pipeline.log'
        }
    }
    
    config_path = temp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

@pytest.fixture
def mock_models(temp_dir):
    """Create mock model checkpoints using actual model instances."""
    # Create model directories
    model_dir = temp_dir / 'models'
    model_dir.mkdir(parents=True)
    
    # Create mock localization model
    loc_dir = model_dir / 'localization'
    loc_dir.mkdir()
    
    # Create UNet model and save its state dict
    unet = UNet(in_channels=3, out_channels=1)
    torch.save({
        'epoch': 0,
        'model_state_dict': unet.state_dict(),
        'optimizer_state_dict': torch.optim.Adam(unet.parameters()).state_dict(),
        'loss': 0.0
    }, loc_dir / 'checkpoint.pth')
    
    # Create mock damage model
    damage_dir = model_dir / 'damage'
    damage_dir.mkdir()
    
    # Create classifier model and save its state dict
    classifier = DamageClassifier(num_classes=4)
    torch.save({
        'epoch': 0,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': torch.optim.Adam(classifier.parameters()).state_dict(),
        'loss': 0.0
    }, damage_dir / 'checkpoint.pth')
    
    return model_dir

@pytest.fixture
def dashboard_server(app, tmp_path):
    """Start dashboard server for testing."""
    results_dir = tmp_path / 'output' / 'classification'
    results_dir.mkdir(parents=True)
    
    # Create mock results with all required fields
    mock_results = {
        'results': [
            {
                'image': 'test_image.png',
                'mask': 'test_mask.png',
                'damage_level': 2,
                'confidence': 0.85,
                'coordinates': {'centroid_lat': 0.0, 'centroid_lon': 0.0},
                'timestamp': datetime.now().isoformat(),
                'building_id': 'test_building_1',
                'area': 100.0,
                'perimeter': 40.0
            }
        ],
        'total': 1,
        'page': 1,
        'page_size': 10,
        'charts': {
            'damage_distribution': {
                'data': [50, 30, 25, 10],
                'labels': ['No Damage', 'Minor', 'Major', 'Destroyed']
            },
            'confidence_histogram': {
                'data': [0.85, 0.75, 0.65, 0.55],
                'bins': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            },
            'timeline': {
                'timestamps': [datetime.now().isoformat()],
                'counts': [115]
            }
        }
    }
    
    # Create mock statistics with all required fields
    mock_stats = {
        'total_detected': 115,
        'damage_levels': {
            '0': {
                'count': 50,
                'avg_confidence': 0.85,
                'min_confidence': 0.7,
                'max_confidence': 0.95,
                'std_confidence': 0.05,
                'confidences': [0.85, 0.9, 0.95]
            },
            '1': {
                'count': 30,
                'avg_confidence': 0.75,
                'min_confidence': 0.6,
                'max_confidence': 0.9,
                'std_confidence': 0.08,
                'confidences': [0.75, 0.8, 0.85]
            },
            '2': {
                'count': 25,
                'avg_confidence': 0.65,
                'min_confidence': 0.5,
                'max_confidence': 0.8,
                'std_confidence': 0.1,
                'confidences': [0.65, 0.7, 0.75]
            },
            '3': {
                'count': 10,
                'avg_confidence': 0.55,
                'min_confidence': 0.4,
                'max_confidence': 0.7,
                'std_confidence': 0.12,
                'confidences': [0.55, 0.6, 0.65]
            }
        },
        'thresholds': {
            '0': 0.7,
            '1': 0.7,
            '2': 0.7,
            '3': 0.7
        }
    }
    
    # Create required directories
    (results_dir / 'images').mkdir(exist_ok=True)
    
    # Write mock data
    with open(results_dir / 'classification_results.json', 'w') as f:
        json.dump(mock_results, f)
    
    with open(results_dir / 'statistics.json', 'w') as f:
        json.dump(mock_stats, f)
    
    # Create a sample test image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(str(results_dir / 'images' / 'test_image.png'), img)
    
    # Start server in a thread
    server = threading.Thread(target=app.run, kwargs={'debug': False, 'port': 5000})
    server.daemon = True
    server.start()
    
    # Wait for server to start
    time.sleep(1)
    
    yield server
    
    # Cleanup
    shutil.rmtree(results_dir)

@pytest.fixture
def test_client(app):
    """Create a test client for the Flask app."""
    return app.test_client()

def test_pipeline_run(temp_dir, sample_data, sample_config, mock_models):
    """Test the pipeline run command."""
    # Set CUDA_VISIBLE_DEVICES to prevent GPU usage in tests
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TESTING'] = '1'
    
    # Create output directories
    output_dir = temp_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Run pipeline
    with pytest.raises(SystemExit) as e:
        cli.main([
            '--config', str(sample_config),
            'run',
            str(sample_data),
            str(output_dir)
        ])
    assert e.value.code == 0
    
    # Verify outputs
    assert (output_dir / 'localization' / 'localization_results.npy').exists()
    assert (output_dir / 'classification' / 'classification_results.json').exists()
    assert (output_dir / 'report' / 'damage_assessment.html').exists()
    assert (output_dir / 'report' / 'damage_assessment.csv').exists()
    assert (output_dir / 'report' / 'damage_assessment.geojson').exists()
    assert (output_dir / 'report' / 'statistics.json').exists()

def test_dashboard_pagination(test_client):
    """Test dashboard pagination."""
    # Test first page
    response = test_client.get('/api/results?page=1&page_size=10')
    assert response.status_code == 200
    data = response.json
    assert len(data['results']) <= 10
    assert 'charts' in data
    assert 'total' in data
    
    # Test second page
    response = test_client.get('/api/results?page=2&page_size=10')
    assert response.status_code == 200
    data = response.json
    assert len(data['results']) <= 10
    
    # Test invalid page parameters
    response = test_client.get('/api/results?page=0&page_size=10')
    assert response.status_code == 400
    
    response = test_client.get('/api/results?page=1&page_size=0')
    assert response.status_code == 400

def test_dashboard_visualizations(test_client):
    """Test dashboard visualizations."""
    response = test_client.get('/api/results?page=1&page_size=10')
    assert response.status_code == 200
    data = response.json
    
    # Check chart data
    assert 'damage_distribution' in data['charts']
    assert 'confidence_histogram' in data['charts']
    assert 'timeline' in data['charts']
    
    # Check chart structure
    damage_chart = data['charts']['damage_distribution']
    assert 'data' in damage_chart
    assert 'labels' in damage_chart
    
    # Test statistics endpoint
    response = test_client.get('/api/statistics')
    assert response.status_code == 200
    stats = response.json
    assert 'total_detected' in stats
    assert 'damage_levels' in stats 