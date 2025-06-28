# xBD Damage Assessment

Satellite image damage assessment with performance metrics and testing.

## Model Performance Metrics

### Localization Model
- **Architecture**: U-Net with encoder-decoder structure
- **Performance**: IoU 0.29, Stable GPU utilization
- **Model File**: `checkpoints/extended/model_epoch_20.pth`
- **Input/Output**: 1024x1024 satellite images → building segmentation masks

### Damage Classification Model  
- **Architecture**: CNN classifier for building damage assessment
- **Overall Accuracy**: **70.2%** (✅ TARGET MET)
- **Mean IoU**: **41.1%** (82% of 50% target)
- **Model File**: `weights/best_damage_model_optimized.pth`
- **Input/Output**: 64x64 building patches → damage class prediction

#### Per-Class Performance:
- **No-damage**: 70.7%
- **Minor-damage**: 64.6% 
- **Major-damage**: 65.2%
- **Destroyed**: 80.8%

## Hardware Requirements

### Minimum System Requirements:
- **GPU**: NVIDIA GPU with CUDA support
- **GPU Memory**: 4GB+ VRAM
- **System RAM**: 8GB+
- **Storage**: 2GB+ available space
- **OS**: Windows 10/11, Linux, or macOS

### Recommended:
- **GPU**: RTX 3060 or better
- **GPU Memory**: 6GB+ VRAM
- **System RAM**: 16GB+

## Software Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

### Core Dependencies:
- Python 3.8+
- PyTorch with CUDA support
- OpenCV (cv2)
- NumPy
- Matplotlib
- Shapely
- scikit-image
- tqdm

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/zak-510/disaster-classifier.git
cd disaster-classifier
```

### 2. Download the Dataset
This project uses the xBD dataset. You will need to download it to train the models or run inference.

1.  Go to the official xView2 website: **[https://xview2.org/](https://xview2.org/)**
2.  Download the dataset. You may need to register.
3.  Create a `Data` directory in the root of the project.
4.  Extract the downloaded dataset and organize it so you have the following structure:
    ```
    Data/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
    ```

### 3. Download Pre-trained Models
The pre-trained models are required to run inference without re-training. They are available on the official GitHub **[Releases page](https://github.com/zak-510/disaster-classifier/releases)** for this repository.

1.  Go to the [Releases page](https://github.com/zak-510/disaster-classifier/releases).
2.  Download the `models.zip` file from the latest release.
3.  Extract the contents into the root of your project directory. This should create or overwrite the `checkpoints` and `weights` directories with the pre-trained model files.

## Quick Start Guide

### 1. Run the Full Inference Pipeline
To run the end-to-end pipeline which performs building localization and damage assessment, run the following command:
```bash
python inference/pipeline_inference.py
```
This script will process test images and save the output visualizations to `test_results/`.

### 2. Run Individual Tests (Optional)

#### Building Localization Test
```bash
python tests/test_localization_inference.py
```
**Expected Output:**
- Generates binary mask visualizations (black background, white buildings)
- Saves PNG files to `test_results/localization/`

#### Damage Classification Test
```bash
python tests/test_damage_inference.py
```
**Expected Output:**
- Generates colored building visualizations on a satellite background
- Saves PNG files to `test_results/damage/`

## Test Image Coverage

The pipeline demonstrates performance across diverse disaster scenarios:

The evaluation set covers ten diverse scenes, including hurricanes, tsunamis, wildfires, and earthquakes (e.g. *palu-tsunami_00000181*, *hurricane-michael_00000437*, *socal-fire_00001400*, *hurricane-florence_00000095*). This provides a broad sanity-check on generalisation without using any training data.

## Directory Structure
```
disaster-classifier/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore             # Files to ignore in git
├── Data/                  # For storing the xBD dataset (not in git)
│   ├── train/
│   └── test/
├── checkpoints/           # For saved localization model checkpoints (not in git)
├── weights/               # For saved damage model weights (not in git)
├── test_results/          # For generated inference outputs 
├── data_processing/       # Scripts for data preprocessing
│   ├── localization_data.py
│   └── damage_data.py
├── models/                # Model architecture definitions
│   ├── __init__.py
│   ├── model.py
│   └── damage_model.py
├── training/              # Scripts for training the models
│   ├── train_localization.py
│   └── train_damage.py
├── inference/             # Scripts for running inference
│   ├── localization_inference.py
│   ├── damage_inference.py
│   └── pipeline_inference.py
└── tests/                 # Test scripts for validation
    ├── test_localization_inference.py
    └── test_damage_inference.py
```

## Model Training (Optional)

If you need to retrain models:

### Localization Model:
```bash
python training/train_localization.py
```

### Damage Classification Model:
```bash
python training/train_damage.py
```

**Note**: Training requires significant computational resources and time.

## Integration & API

### Core Inference Functions:

#### Building Localization:
```python
from inference.localization_inference import run_localization_inference
results = run_localization_inference(image_path)
```

#### Damage Classification:
```python
from inference.damage_inference import run_damage_inference
results = run_damage_inference(image_path)
```

#### Full Pipeline:
```python
from inference.pipeline_inference import run_full_pipeline
results = run_full_pipeline(image_path)
```

## Troubleshooting

### Common Issues:

**CUDA Out of Memory:**
- Reduce batch size in training scripts
- Ensure sufficient GPU memory (4GB+ required)

**Model File Not Found:**
- Verify model files exist in correct locations
- Check file paths in inference scripts

**Import Errors:**
- Install all dependencies: `pip install -r requirements.txt`
- Ensure PyTorch CUDA version matches your system

**Performance Issues:**
- Use GPU for inference (CPU inference is significantly slower)
- Verify CUDA is properly installed and accessible

## Performance Monitoring

The pipeline includes comprehensive logging and performance metrics:
- Model loading confirmation
- Processing time per image
- Building detection counts
- Accuracy metrics per test image

## Production Deployment

This pipeline is validated for production use with:
- Stable inference performance
- Comprehensive error handling
- Clean visualization outputs
- Professional logging format
- Memory-efficient processing

## Support

For technical support or questions about deployment:
1. Verify all dependencies are installed correctly
2. Run both test scripts to confirm functionality
3. Check generated outputs in `test_results/` directory
4. Review logs for any error messages

