"""Pipeline orchestration module."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import click
from datetime import datetime

from ..steps.localize import run_localization
from ..steps.classify import run_classification
from ..steps.report import generate_report
from ..config.validation import PipelineConfig

logger = logging.getLogger(__name__)

@click.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option('--config', type=click.Path(exists=True, path_type=Path), help='Path to config file')
@click.option('--localization-model', type=click.Path(exists=True, path_type=Path), help='Path to localization model')
@click.option('--damage-model', type=click.Path(exists=True, path_type=Path), help='Path to damage model')
@click.option('--batch-size', type=int, help='Batch size for inference')
@click.option('--threshold', type=float, help='Detection threshold')
@click.option('--min-confidence', type=float, help='Minimum confidence for damage classification')
def run(
    input_path: Path,
    output_path: Path,
    config: Optional[Path] = None,
    localization_model: Optional[Path] = None,
    damage_model: Optional[Path] = None,
    batch_size: Optional[int] = None,
    threshold: Optional[float] = None,
    min_confidence: Optional[float] = None
) -> None:
    """Run the complete xBD pipeline."""
    start_time = datetime.now()
    logger.info(f"Starting pipeline at {start_time}")
    
    # Load and validate config
    pipeline_config = PipelineConfig.from_yaml(config) if config else PipelineConfig()
    
    # Override config with CLI options
    if batch_size:
        pipeline_config.inference.batch_size = batch_size
    if threshold:
        pipeline_config.inference.threshold = threshold
    if min_confidence:
        pipeline_config.inference.min_confidence = min_confidence
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    localization_output = output_path / 'localization'
    classification_output = output_path / 'classification'
    report_output = output_path / 'report'
    
    # Run localization
    logger.info("Starting localization step")
    run_localization(
        input_path=input_path,
        output_path=localization_output,
        model_path=localization_model or pipeline_config.models.localization.checkpoint,
        batch_size=pipeline_config.inference.batch_size,
        threshold=pipeline_config.inference.threshold,
        config=pipeline_config.dict()
    )
    
    # Run classification
    logger.info("Starting classification step")
    run_classification(
        input_path=localization_output,
        output_path=classification_output,
        model_path=damage_model or pipeline_config.models.damage.checkpoint,
        batch_size=pipeline_config.inference.batch_size,
        min_confidence=pipeline_config.inference.min_confidence,
        config=pipeline_config.dict()
    )
    
    # Generate report
    logger.info("Starting report generation")
    generate_report(
        input_path=classification_output,
        output_path=report_output,
        config=pipeline_config.dict()
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Pipeline finished in {duration}")
    logger.info(f"Results saved to {output_path}") 