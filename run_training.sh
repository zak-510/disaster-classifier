#!/bin/bash

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# Create output directory
OUTPUT_DIR="xbd_pipeline_output/localization_model"
mkdir -p $OUTPUT_DIR

# Run training with our improved settings
python train_localization.py \
    --data_dir Data \
    --output_dir $OUTPUT_DIR \
    --epochs 1 \
    --batch_size 4 \
    --num_workers 2 