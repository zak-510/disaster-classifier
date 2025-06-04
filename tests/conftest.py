import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

@pytest.fixture
def test_data_dir():
    return Path("tests/test_data")

@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

@pytest.fixture
def sample_mask():
    return np.random.randint(0, 2, (512, 512), dtype=np.uint8)

@pytest.fixture
def sample_config():
    return {
        "model": {
            "architecture": "unet",
            "encoder": "resnet34",
            "num_classes": 1,
            "learning_rate": 0.001
        },
        "data": {
            "train_dir": "Data/train",
            "val_dir": "Data/val",
            "batch_size": 8,
            "num_workers": 4
        },
        "training": {
            "epochs": 100,
            "early_stopping_patience": 10,
            "checkpoint_dir": "output/checkpoints"
        }
    }

@pytest.fixture
def cuda_available():
    return torch.cuda.is_available()

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    os.environ["TESTING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU during testing
    yield
    os.environ.pop("TESTING", None) 