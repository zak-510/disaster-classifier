# Troubleshooting Guide

This guide covers common issues and their solutions when working with the xBD pipeline.

## Environment Setup

### CUDA/GPU Issues

1. **CUDA not found**
   ```
   RuntimeError: CUDA not available
   ```
   **Solution:**
   - Check if NVIDIA drivers are installed: `nvidia-smi`
   - Verify CUDA installation: `nvcc --version`
   - Ensure PyTorch is installed with CUDA support:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
     ```

2. **Out of GPU Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution:**
   - Reduce batch size in config
   - Use gradient accumulation
   - Enable mixed precision training
   - Use smaller model architecture

### Python Environment

1. **Package Version Conflicts**
   ```
   ImportError: cannot import name X from Y
   ```
   **Solution:**
   - Create fresh conda environment
   - Install exact versions from requirements.txt
   - Check package compatibility matrix

2. **Missing Dependencies**
   ```
   ModuleNotFoundError: No module named X
   ```
   **Solution:**
   - Run `pip install -r requirements.txt`
   - Check if package is in requirements.txt
   - Verify Python path

## Data Processing

1. **Memory Issues with Large Images**
   ```
   MemoryError: Unable to allocate X bytes
   ```
   **Solution:**
   - Use smaller image size
   - Process in smaller batches
   - Enable memory mapping
   - Use data streaming

2. **Corrupted Data**
   ```
   ValueError: Invalid image data
   ```
   **Solution:**
   - Verify data integrity
   - Check file permissions
   - Validate image format
   - Use data validation script

## Training

1. **Loss Not Decreasing**
   ```
   Loss stuck at X
   ```
   **Solution:**
   - Check learning rate
   - Verify data preprocessing
   - Check model architecture
   - Monitor gradients
   - Try different optimizer

2. **Overfitting**
   ```
   Training accuracy >> Validation accuracy
   ```
   **Solution:**
   - Increase regularization
   - Add more augmentation
   - Use early stopping
   - Try different model architecture
   - Collect more data

3. **NaN Loss**
   ```
   Loss is NaN
   ```
   **Solution:**
   - Check for division by zero
   - Verify input normalization
   - Check learning rate
   - Use gradient clipping
   - Check for corrupted data

## Inference

1. **Slow Inference**
   ```
   Inference taking too long
   ```
   **Solution:**
   - Use batch processing
   - Enable CUDA optimization
   - Use model quantization
   - Try ONNX export
   - Use TensorRT

2. **Incorrect Predictions**
   ```
   Poor prediction quality
   ```
   **Solution:**
   - Check model checkpoint
   - Verify preprocessing
   - Check confidence thresholds
   - Validate input data
   - Retrain with more data

## Docker Issues

1. **Container Build Fails**
   ```
   Error building Docker image
   ```
   **Solution:**
   - Check Dockerfile syntax
   - Verify dependencies
   - Check disk space
   - Update Docker

2. **GPU Not Available in Container**
   ```
   CUDA not available in container
   ```
   **Solution:**
   - Install NVIDIA Container Toolkit
   - Use `--gpus all` flag
   - Check Docker runtime
   - Verify GPU passthrough

## Common Commands

### Environment
```bash
# Create conda environment
conda create -n xbd python=3.8
conda activate xbd

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Data Processing
```bash
# Validate data
python src/utils/validate_data.py --data_dir Data

# Generate masks
python mask_polygons.py --input Data --output masks
```

### Training
```bash
# Train localization
python train_localization.py --config src/configs/model_config.yaml

# Train classification
python damage_classification.py --config src/configs/model_config.yaml
```

### Inference
```bash
# Run inference
python run_inference.py --input_dir test_images --output_dir predictions
```

## Getting Help

If you encounter an issue not covered here:

1. Check the [GitHub Issues](https://github.com/your-repo/issues)
2. Search the [Discussions](https://github.com/your-repo/discussions)
3. Create a new issue with:
   - Error message
   - Environment details
   - Steps to reproduce
   - Expected vs actual behavior 