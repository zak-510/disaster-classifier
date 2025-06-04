@echo off
REM inference.bat
REM Runs the full inference pipeline from pre/post-disaster images to final output
REM Usage: inference.bat <output_dir> [--test_mode]

set OUTPUT_DIR=%1
set TEST_MODE=false
if "%2"=="--test_mode" set TEST_MODE=true

REM Placeholder: Run localization model inference
python run_localization_inference.py --model_dir "%OUTPUT_DIR%\localization_model" --data_dir "%OUTPUT_DIR%\spacenet_structure\val\images" --output_dir "%OUTPUT_DIR%\inference\localization" %TEST_MODE%

REM Placeholder: Run damage classification inference
python run_damage_inference.py --model_dir "%OUTPUT_DIR%\damage_model" --data_dir "%OUTPUT_DIR%\inference\localization" --output_dir "%OUTPUT_DIR%\inference\damage" %TEST_MODE%

echo [Done] Inference pipeline complete. 