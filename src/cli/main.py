"""Main CLI entry point for xBD pipeline."""

import click
from pathlib import Path
import yaml
from typing import Optional

from . import setup_logging, validate_path, logger

@click.group()
@click.option('--config', type=click.Path(exists=True), help='Path to config file')
@click.option('--log-file', type=click.Path(), help='Path to log file')
@click.pass_context
def cli(ctx, config: Optional[str], log_file: Optional[str]):
    """xBD Pipeline CLI."""
    ctx.ensure_object(dict)
    
    # Setup logging
    if log_file:
        setup_logging(Path(log_file))
    
    # Load config if provided
    if config:
        with open(config) as f:
            ctx.obj['config'] = yaml.safe_load(f)

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--model-path', type=click.Path(exists=True), required=True)
@click.option('--batch-size', type=int, default=16)
@click.option('--threshold', type=float, default=0.5)
@click.pass_context
def localize(ctx, input_dir: str, output_dir: str, model_path: str, batch_size: int, threshold: float):
    """Run building localization on input images."""
    from ..steps.localize import run_localization
    
    input_path = validate_path(input_dir)
    output_path = validate_path(output_dir, must_exist=False)
    model_path = validate_path(model_path)
    
    try:
        run_localization(
            input_path=input_path,
            output_path=output_path,
            model_path=model_path,
            batch_size=batch_size,
            threshold=threshold,
            config=ctx.obj.get('config', {})
        )
    except Exception as e:
        logger.error(f"Localization failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--model-path', type=click.Path(exists=True), required=True)
@click.option('--batch-size', type=int, default=16)
@click.option('--min-confidence', type=float, default=0.7)
@click.pass_context
def classify(ctx, input_dir: str, output_dir: str, model_path: str, batch_size: int, min_confidence: float):
    """Run damage classification on localized buildings."""
    from ..steps.classify import run_classification
    
    input_path = validate_path(input_dir)
    output_path = validate_path(output_dir, must_exist=False)
    model_path = validate_path(model_path)
    
    try:
        run_classification(
            input_path=input_path,
            output_path=output_path,
            model_path=model_path,
            batch_size=batch_size,
            min_confidence=min_confidence,
            config=ctx.obj.get('config', {})
        )
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.pass_context
def report(ctx, input_dir: str, output_dir: str):
    """Generate HTML report from classification results."""
    from ..steps.report import generate_report
    
    input_path = validate_path(input_dir)
    output_path = validate_path(output_dir, must_exist=False)
    
    try:
        generate_report(
            input_path=input_path,
            output_path=output_path,
            config=ctx.obj.get('config', {})
        )
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.pass_context
def run(ctx, input_dir: str, output_dir: str):
    """Run the complete xBD pipeline."""
    from ..run_pipeline import run_pipeline

    input_path = validate_path(input_dir)
    output_path = validate_path(output_dir, must_exist=False)

    try:
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary config file
        config = {
            'data': {
                'input_dir': str(input_path),
                'batch_size': 8,
                'image_size': [1024, 1024]
            },
            'models': {
                'localization': {
                    'checkpoint': str(input_path / 'models/localization/checkpoint.pth'),
                    'output_name': 'localization_model.pth'
                },
                'damage': {
                    'checkpoint': str(input_path / 'models/damage/checkpoint.pth'),
                    'output_name': 'damage_model.pth'
                }
            }
        }
        
        config_path = output_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        run_pipeline(config_path=config_path, output_dir=output_path)
    except Exception as e:
        logger.error(f"Pipeline run failed: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli(obj={}) 