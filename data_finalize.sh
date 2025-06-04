#!/bin/bash
# data_finalize.sh
# Sets up SpaceNet model directory structure, creates train/val splits, and runs compute_mean.py
# Usage: bash data_finalize.sh <input_dir> <output_dir> [--test_mode]

set -e

INPUT_DIR=$1
OUTPUT_DIR=$2
TEST_MODE=false
if [[ "$3" == "--test_mode" ]]; then
  TEST_MODE=true
fi

mkdir -p "$OUTPUT_DIR"

# Copy images and masks to SpaceNet structure
for SPLIT in train val; do
  mkdir -p "$OUTPUT_DIR/$SPLIT/images"
  mkdir -p "$OUTPUT_DIR/$SPLIT/labels"
  mkdir -p "$OUTPUT_DIR/$SPLIT/masks"

done

# For test mode, use only a small subset
if $TEST_MODE; then
  echo "[Test mode] Using a small subset for train/val."
  # Copy only 2 images per split
  for SPLIT in train val; do
    find "$INPUT_DIR/$SPLIT" -type f -name '*.png' | head -n 2 | while read IMG; do
      cp "$IMG" "$OUTPUT_DIR/$SPLIT/images/"
    done
  done
else
  # Copy all images
  for SPLIT in train val; do
    find "$INPUT_DIR/$SPLIT" -type f -name '*.png' | while read IMG; do
      cp "$IMG" "$OUTPUT_DIR/$SPLIT/images/"
    done
  done
fi

# Run compute_mean.py (placeholder, to be implemented)
echo "[Info] Running compute_mean.py (not implemented in this script)"
# python compute_mean.py --data_dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/mean.npy"

echo "[Done] SpaceNet structure ready." 