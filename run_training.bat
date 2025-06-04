@echo off

:: Set environment variables for better GPU memory management
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set CUDA_LAUNCH_BLOCKING=1

:: Create output directory
set OUTPUT_DIR=xbd_pipeline_output\localization_model
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

:: Run training with our improved settings
python train_localization.py ^
    --data_dir Data ^
    --output_dir %OUTPUT_DIR% ^
    --epochs 1 ^
    --batch_size 4 ^
    --num_workers 2

pause 