import os
import pytest
from pathlib import Path
from src.config import Config, ConfigError

def test_config_from_yaml(sample_config, tmp_path):
    # Write sample config to temp file
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        import yaml
        yaml.dump(sample_config, f)
    
    # Test loading config
    config = Config.from_yaml(config_path)
    assert config.model.architecture == "unet"
    assert config.data.batch_size == 8

def test_config_validation():
    with pytest.raises(ConfigError):
        Config(
            model={"architecture": "invalid"},
            data={"batch_size": -1},
            training={"epochs": 0}
        )

def test_config_env_override(sample_config, monkeypatch):
    monkeypatch.setenv("XBD_BATCH_SIZE", "16")
    monkeypatch.setenv("XBD_LEARNING_RATE", "0.01")
    
    config = Config(**sample_config)
    assert config.data.batch_size == 16
    assert config.model.learning_rate == 0.01

def test_config_cli_override(sample_config):
    cli_args = ["--batch-size", "32", "--learning-rate", "0.1"]
    config = Config.from_cli(cli_args, defaults=sample_config)
    assert config.data.batch_size == 32
    assert config.model.learning_rate == 0.1

def test_config_save_load(sample_config, tmp_path):
    config = Config(**sample_config)
    save_path = tmp_path / "config.yaml"
    config.save(save_path)
    
    loaded = Config.from_yaml(save_path)
    assert loaded.model.dict() == config.model.dict()
    assert loaded.data.dict() == config.data.dict() 