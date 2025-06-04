import os
import argparse
import subprocess
import sys
from pathlib import Path

"""
run_pipeline.py
==============
Runs the complete xBD building damage assessment pipeline.

Usage:
    python run_pipeline.py --raw_data_dir Data --output_dir xbd_pipeline_output [--test_mode]

Arguments:
    --raw_data_dir: Path to raw xBD data directory (with train/images and train/labels).
    --output_dir: Path to save pipeline outputs (models, predictions, etc.).
    --test_mode: (Optional) If set, runs on a small subset for quick testing.
"""

def run_command(cmd, description):
    print(f"\n[Info] {description}")
    print(f"[Info] Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"[Success] {description} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"[Error] {description} failed with error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the xBD pipeline.")
    parser.add_argument('--raw_data_dir', type=str, required=True, help='Path to raw xBD data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save outputs')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with small subset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training (default: 4)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train localization model
    print("[Info] Training building localization model")
    localization_cmd = f"python train_localization.py --data_dir {args.raw_data_dir} --output_dir {os.path.join(args.output_dir, 'localization_model')} --batch_size {args.batch_size}"
    if args.test_mode:
        localization_cmd += " --test_mode"
    print(f"[Info] Running: {localization_cmd}")
    if os.system(localization_cmd) != 0:
        print("[Error] Training building localization model failed")
        return
    print("[Success] Training building localization model completed successfully")

    # Train damage classification model
    print("[Info] Training damage classification model")
    damage_cmd = f"python damage_classification.py --data_dir {args.raw_data_dir} --output_dir {os.path.join(args.output_dir, 'damage_model')} --batch_size {args.batch_size}"
    if args.test_mode:
        damage_cmd += " --test_mode"
    print(f"[Info] Running: {damage_cmd}")
    if os.system(damage_cmd) != 0:
        print("[Error] Training damage classification model failed")
        return
    print("[Success] Training damage classification model completed successfully")

    # 3. Run inference on test set
    inference_cmd = f"python run_inference.py --data_dir {args.raw_data_dir} --output_dir {os.path.join(args.output_dir, 'predictions')} --localization_model {os.path.join(args.output_dir, 'localization_model', 'model_epoch10.pth')} --damage_model {os.path.join(args.output_dir, 'damage_model', 'model_epoch10.pth')}"
    if args.test_mode:
        inference_cmd += " --test_mode"
    run_command(inference_cmd, "Running inference on test set")

    print("\n[Success] Pipeline completed successfully!")

if __name__ == "__main__":
    main() 