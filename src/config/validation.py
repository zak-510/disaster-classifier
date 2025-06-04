"""Configuration validation module."""

from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import yaml
import os

class LocalizationModelConfig(BaseModel):
    """Localization model configuration."""
    checkpoint: Path
    architecture: str = "resnet50"
    pretrained: bool = True

class DamageLevelConfig(BaseModel):
    """Damage level configuration."""
    name: str
    min_confidence: float = Field(ge=0.0, le=1.0)

class DamageModelConfig(BaseModel):
    """Damage model configuration."""
    checkpoint: Path
    architecture: str = "resnet50"
    pretrained: bool = True
    damage_levels: Dict[str, DamageLevelConfig]

class ModelConfig(BaseModel):
    """Model configuration."""
    localization: LocalizationModelConfig
    damage: DamageModelConfig

class InferenceConfig(BaseModel):
    """Inference configuration."""
    threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    batch_size: int = Field(gt=0, default=16)
    device: str = 'cuda'
    save_debug: bool = True
    debug_dir: str = 'debug'
    output_format: str = 'geojson'
    min_confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    min_area: float = Field(gt=0, default=100.0)

class DataConfig(BaseModel):
    """Data configuration."""
    input_dir: Path
    image_size: tuple[int, int] = (1024, 1024)
    batch_size: int = 8
    num_workers: int = 4

    @validator('input_dir')
    def validate_input_dir(cls, v):
        """Validate input directory exists."""
        if not os.getenv('TESTING') and not v.exists():
            raise ValueError(f'Input directory {v} does not exist')
        return v

class PipelineConfig(BaseModel):
    """Pipeline configuration."""
    data: DataConfig
    models: ModelConfig
    inference: InferenceConfig = InferenceConfig()

    @classmethod
    def from_yaml(cls, path: Optional[Path]) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        if path is None:
            return cls(
                data=DataConfig(input_dir=Path(".")),
                models=ModelConfig(
                    localization=LocalizationModelConfig(checkpoint=Path(".")),
                    damage=DamageModelConfig(
                        checkpoint=Path("."),
                        damage_levels={
                            "0": DamageLevelConfig(name="no_damage"),
                            "1": DamageLevelConfig(name="minor_damage"),
                            "2": DamageLevelConfig(name="major_damage"),
                            "3": DamageLevelConfig(name="destroyed")
                        }
                    )
                )
            )
        
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @validator('data')
    def validate_data_paths(cls, v: DataConfig) -> DataConfig:
        """Validate data paths exist."""
        if not v.input_dir.exists():
            raise ValueError(f"Input directory {v.input_dir} does not exist")
        return v

    @validator('models')
    def validate_model_paths(cls, v: ModelConfig) -> ModelConfig:
        """Validate model paths exist."""
        if not v.localization.checkpoint.exists():
            raise ValueError(f"Localization model {v.localization.checkpoint} does not exist")
        if not v.damage.checkpoint.exists():
            raise ValueError(f"Damage model {v.damage.checkpoint} does not exist")
        return v

class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(levelname)s - %(message)s'
    file: Optional[str] = None

class Config(BaseModel):
    """Pipeline configuration."""
    data: DataConfig
    models: Dict[str, ModelConfig]
    inference: InferenceConfig
    logging: LoggingConfig

def validate_config(config: Dict[str, Any], test_mode: bool = False) -> bool:
    """Validate configuration."""
    try:
        if test_mode:
            os.environ['TESTING'] = '1'
        Config(**config)
        return True
    except Exception as e:
        print(f'Configuration validation failed: {e}')
        return False
    finally:
        if test_mode:
            os.environ.pop('TESTING', None) 