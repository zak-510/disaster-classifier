# xBD Pipeline

A machine learning pipeline for disaster damage assessment using the xBD dataset.

## Overview

This pipeline processes satellite imagery to:
1. Detect buildings (localization)
2. Classify damage levels (classification)

## Quick Start

### Using Docker (Recommended)

1. Install [Docker](https://docs.docker.com/get-docker/) and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

2. Set environment variables:
```bash
export WANDB_API_KEY=your_key_here  # Optional, for W&B logging
```

3. Run the pipeline:
```bash
# For training
docker-compose up train

# For inference
docker-compose up inference

# For monitoring (optional)
docker-compose up tensorboard mlflow
```

### Manual Setup

1. Create conda environment:
```bash
conda create -n xbd python=3.10
conda activate xbd
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the pipeline:
```bash
# Training
python run_training.py --config configs/train.yaml

# Inference
python run_inference.py --config configs/inference.yaml
```

## Project Structure

```
xbd-pipeline/
├── configs/           # Configuration files
├── Data/              # Dataset directory
├── src/              # Source code
│   ├── models/       # Model architectures
│   ├── utils/        # Utility functions
│   └── steps/        # Pipeline steps
├── tests/            # Unit tests
└── output/           # Model outputs and artifacts
```

## Configuration

All configuration is centralized in YAML files under `configs/`. Override via:
- Environment variables (prefix: `XBD_`)
- CLI arguments
- Config file

Example:
```bash
XBD_BATCH_SIZE=32 python run_training.py --learning-rate 0.001
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py
```

### Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type checking
mypy src tests
```

## Experiment Tracking

The pipeline uses MLflow for experiment tracking. Access the UI at http://localhost:5000 when running via Docker.

Key metrics tracked:
- Training/validation loss
- IoU (localization)
- F1 score (classification)
- Learning rate
- GPU utilization

## Troubleshooting

Common issues:

1. CUDA out of memory:
   - Reduce batch size in config
   - Use gradient accumulation
   - Use mixed precision training

2. Dataset errors:
   - Verify data structure matches expected format
   - Check image dimensions and formats
   - Validate label files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details 