import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd: str, cwd: Optional[Path] = None) -> bool:
    """Run a command and handle errors consistently."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Command output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with error: {e.stderr}")
        return False

def validate_inputs(config: Dict[str, Any]) -> bool:
    """Validate input directories and files."""
    required_paths = [
        config['data']['input_dir'],
        config['models']['localization']['checkpoint'],
        config['models']['damage']['checkpoint']
    ]
    
    for path in required_paths:
        if not Path(path).exists():
            logger.error(f"Required path does not exist: {path}")
            return False
    
    return True

def run_pipeline(config_path: Path, output_dir: Path) -> bool:
    """Run the complete xBD pipeline."""
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validate inputs
    if not validate_inputs(config):
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Train localization model
    logger.info("Training localization model...")
    loc_cmd = (
        f"python train_localization.py "
        f"--config {config_path} "
        f"--data_dir {config['data']['input_dir']} "
        f"--output_dir {output_dir / 'localization'} "
        f"--checkpoint {config['models']['localization']['checkpoint']}"
    )
    if not run_command(loc_cmd):
        logger.error("Localization model training failed")
        return False
    
    # Step 2: Train damage classification model
    logger.info("Training damage classification model...")
    damage_cmd = (
        f"python damage_classification.py "
        f"--config {config_path} "
        f"--data_dir {config['data']['input_dir']} "
        f"--output_dir {output_dir / 'damage'} "
        f"--checkpoint {config['models']['damage']['checkpoint']}"
    )
    if not run_command(damage_cmd):
        logger.error("Damage classification model training failed")
        return False
    
    # Step 3: Run inference
    logger.info("Running inference...")
    inference_cmd = (
        f"python run_inference.py "
        f"--input_dir {config['data']['input_dir']} "
        f"--output_dir {output_dir / 'inference'} "
        f"--localization_model {output_dir / 'localization' / config['models']['localization']['output_name']} "
        f"--damage_model {output_dir / 'damage' / config['models']['damage']['output_name']} "
        f"--threshold {config['inference']['threshold']} "
        f"--batch_size {config['inference']['batch_size']}"
    )
    if not run_command(inference_cmd):
        logger.error("Inference failed")
        return False
    
    logger.info("Pipeline completed successfully")
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run xBD pipeline')
    parser.add_argument('--config', type=Path, required=True, help='Path to configuration file')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    if not run_pipeline(args.config, args.output_dir):
        sys.exit(1) 