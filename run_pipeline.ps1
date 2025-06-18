# xBD Building Detection Pipeline (Windows PowerShell)
# Complete workflow from raw data to evaluation

$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting xBD Building Detection Pipeline" -ForegroundColor Green
Write-Host "==============================================`n"

# Check prerequisites
Write-Host "📋 Checking prerequisites..." -ForegroundColor Yellow

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check NVIDIA GPU
try {
    nvidia-smi | Out-Null
    Write-Host "✅ NVIDIA GPU detected" -ForegroundColor Green
} catch {
    Write-Host "⚠️  NVIDIA GPU not detected. Pipeline will run slower on CPU." -ForegroundColor Yellow
}

# Check required files
$requiredFiles = @(
    "config.py",
    "models/localization_model.py", 
    "training/train_localization.py",
    "validate_localization.py"
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Host "❌ Required file missing: $file" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ Prerequisites check passed`n" -ForegroundColor Green

# Step 1: Data Preprocessing
Write-Host "📊 Step 1: Data Preprocessing" -ForegroundColor Cyan
Write-Host "----------------------------"

if (-not (Test-Path "data/images_processed.npy") -or -not (Test-Path "data/masks_processed.npy")) {
    Write-Host "🔄 Processing raw data..."
    python mask_polygons.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Data preprocessing failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Data preprocessing complete" -ForegroundColor Green
} else {
    Write-Host "✅ Processed data already exists, skipping preprocessing" -ForegroundColor Green
}

# Step 2: Model Training
Write-Host "`n🧠 Step 2: Model Training" -ForegroundColor Cyan
Write-Host "------------------------"

if (-not (Test-Path "weights/localization_best.pth")) {
    Write-Host "🔄 Training localization model..."
    python training/train_localization.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Model training failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Model training complete" -ForegroundColor Green
} else {
    Write-Host "✅ Trained model already exists, skipping training" -ForegroundColor Green
    $retrain = Read-Host "Do you want to retrain the model? (y/N)"
    if ($retrain -eq "y" -or $retrain -eq "Y") {
        Write-Host "🔄 Retraining model..."
        python training/train_localization.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Model retraining failed" -ForegroundColor Red
            exit 1
        }
        Write-Host "✅ Model retraining complete" -ForegroundColor Green
    }
}

# Step 3: Model Validation  
Write-Host "`n🎯 Step 3: Model Validation" -ForegroundColor Cyan
Write-Host "-------------------------"
Write-Host "🔄 Running validation..."

python validate_localization.py
$validationResult = $LASTEXITCODE

if ($validationResult -eq 0) {
    Write-Host "✅ Validation PASSED - Model ready for deployment" -ForegroundColor Green
} else {
    Write-Host "❌ Validation FAILED - Model needs improvement" -ForegroundColor Red
    Write-Host "📊 Check outputs/ directory for validation visualizations"
}

# Step 4: Generate Evaluation Report
Write-Host "`n📈 Step 4: Evaluation Report" -ForegroundColor Cyan
Write-Host "---------------------------"
Write-Host "🔄 Generating evaluation report..."

# Create a simple Python script to generate the report
$pythonCode = @"
import os
import numpy as np
from datetime import datetime

# Create evaluation directory
os.makedirs('evaluation_results', exist_ok=True)

# Get validation result from command line argument
import sys
validation_passed = len(sys.argv) > 1 and sys.argv[1] == '0'

# Generate report
report = f'''# xBD Building Detection Pipeline - Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Pipeline Status: {'✅ COMPLETED' if validation_passed else '⚠️ COMPLETED WITH ISSUES'}

### Architecture Summary
- **Model:** Simplified U-Net (16→128 channels)
- **Loss Function:** FocalDiceLoss (Dice + Focal Loss)
- **Training Data:** 100 processed images
- **Validation:** 10 test samples

### Performance Metrics
- **Validation Result:** {'PASSED' if validation_passed else 'FAILED'}
- **Training Epochs:** 30
- **Best Model:** weights/localization_best.pth

### Output Files
- **Processed Data:** data/images_processed.npy, data/masks_processed.npy
- **Model Weights:** weights/localization_best.pth, weights/localization_final.pth
- **Validation Images:** outputs/validation_*.png

### Next Steps
1. Review validation visualizations in outputs/ directory
2. If Mean IoU < 0.5, consider data augmentation or transfer learning
3. Deploy model using inference/ scripts

### File Structure
```
xbd-pipeline/
├── data/                 # Processed training data
├── weights/              # Trained model weights
├── outputs/              # Validation visualizations
├── models/               # Model architecture
├── training/             # Training scripts
└── evaluation_results/   # This report
```
'''

with open('evaluation_results/evaluation_report.md', 'w') as f:
    f.write(report)

print('✅ Evaluation report generated: evaluation_results/evaluation_report.md')
"@

python -c $pythonCode $validationResult

Write-Host "`n🎉 Pipeline Execution Complete!" -ForegroundColor Green
Write-Host "==============================="
Write-Host "📁 Check these directories for results:"
Write-Host "   - outputs/: Validation visualizations"  
Write-Host "   - weights/: Trained model files"
Write-Host "   - evaluation_results/: Comprehensive report"
Write-Host "`n📊 To view validation results:"
Write-Host "   Get-Content evaluation_results/evaluation_report.md"

if ($validationResult -ne 0) {
    Write-Host "`n⚠️  Pipeline completed with validation issues" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "`n✅ Pipeline completed successfully!" -ForegroundColor Green
    exit 0
}