#!/bin/bash
# inference.sh
# Runs the full inference pipeline from pre/post-disaster images to final output
# Usage: bash inference.sh <output_dir> [--test_mode]

set -e

OUTPUT_DIR=$1
TEST_MODE=false
if [[ "$2" == "--test_mode" ]]; then
  TEST_MODE=true
fi

# Placeholder: Run localization model inference
python run_localization_inference.py --model_dir "$OUTPUT_DIR/localization_model" --data_dir "$OUTPUT_DIR/spacenet_structure/val/images" --output_dir "$OUTPUT_DIR/inference/localization" ${TEST_MODE:+--test_mode}

# Placeholder: Run damage classification inference
python run_damage_inference.py --model_dir "$OUTPUT_DIR/damage_model" --data_dir "$OUTPUT_DIR/inference/localization" --output_dir "$OUTPUT_DIR/inference/damage" ${TEST_MODE:+--test_mode}

echo "[Done] Inference pipeline complete." 