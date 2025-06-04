import os
import json
import tempfile
from pathlib import Path
import numpy as np
import cv2
import yaml
from unittest.mock import MagicMock

# Mock DamageClassifier class
class DamageClassifier:
    def __init__(self, *args, **kwargs):
        pass
    
    def predict(self, *args, **kwargs):
        # Return mock predictions with confidence scores
        return {
            'predictions': np.array([0, 1, 2, 3, 4]),  # Mock damage levels
            'confidences': np.array([0.95, 0.85, 0.75, 0.65, 0.55])  # Mock confidence scores
        }

# Import the report generator after mocking
from src.utils.report_generator import HTMLReportGenerator

def create_sample_data(output_dir: Path):
    """Create sample data for testing."""
    # Create disaster directory
    disaster_dir = output_dir / "sample_disaster"
    disaster_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images directory
    images_dir = disaster_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create sample pre/post images
    for img_type in ["pre", "post"]:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"{img_type}_disaster.png"), img)
    
    return disaster_dir

def create_sample_config(output_dir: Path):
    """Create sample configuration file."""
    config = {
        'model': {
            'features': 64,
            'damage_levels': ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
        }
    }
    
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

def create_sample_models(output_dir: Path):
    """Create sample model checkpoints."""
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Create empty model files
    (models_dir / "localization_model.pt").touch()
    (models_dir / "classification_model.pt").touch()
    
    return models_dir

def test_report_generation():
    """Test the report generation functionality."""
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample data
        disaster_dir = create_sample_data(temp_path)
        
        # Create sample config
        config_path = create_sample_config(temp_path)
        
        # Create sample models
        models_dir = create_sample_models(temp_path)
        
        # Create sample statistics
        stats = {
            'total_detected': 115,
            'damage_levels': {
                0: {
                    'count': 40,
                    'avg_confidence': 0.85,
                    'min_confidence': 0.80,
                    'max_confidence': 0.90,
                    'std_confidence': 0.02,
                    'confidences': np.random.uniform(0.8, 0.9, 40).tolist()
                },
                1: {
                    'count': 30,
                    'avg_confidence': 0.75,
                    'min_confidence': 0.70,
                    'max_confidence': 0.80,
                    'std_confidence': 0.03,
                    'confidences': np.random.uniform(0.7, 0.8, 30).tolist()
                },
                2: {
                    'count': 20,
                    'avg_confidence': 0.65,
                    'min_confidence': 0.60,
                    'max_confidence': 0.70,
                    'std_confidence': 0.03,
                    'confidences': np.random.uniform(0.6, 0.7, 20).tolist()
                },
                3: {
                    'count': 10,
                    'avg_confidence': 0.55,
                    'min_confidence': 0.50,
                    'max_confidence': 0.60,
                    'std_confidence': 0.03,
                    'confidences': np.random.uniform(0.5, 0.6, 10).tolist()
                }
            },
            'thresholds': {
                0: 0.80,
                1: 0.70,
                2: 0.60,
                3: 0.50
            }
        }
        
        # Create sample visualizations
        vis_dir = disaster_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        for i in range(5):
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(vis_dir / f"damage_vis_{i}.png"), img)
        
        # Initialize report generator
        report_generator = HTMLReportGenerator(
            output_dir=temp_path,
            disaster_name="sample_disaster"
        )
        
        # Generate report
        report_path = report_generator.generate_report(
            stats=stats,
            example_images=list(vis_dir.glob("*.png"))
        )
        
        # Verify report exists
        assert report_path.exists(), "Report file was not created"
        
        # Verify report content
        with open(report_path) as f:
            content = f.read()
            assert "Damage Assessment Report" in content
            assert "sample_disaster" in content
            assert "Plotly.newPlot" in content
        
        return report_path

if __name__ == '__main__':
    test_report_generation() 