# Hugging Face Integration for 2D to 3D Model Training

This directory contains all the files needed to train your 2D to 3D reconstruction model using Hugging Face datasets.

## Directory Structure

```
train_with_huggingface/
├── README.md                           # This file
├── README_HuggingFace.md              # Detailed Hugging Face usage guide
├── huggingface_dataset_loader.py      # Core dataset loading functionality
├── train_with_huggingface.py          # Main training script
├── test_hf_integration.py             # Integration testing script
└── install_huggingface.py             # Automated installation script
```

## Quick Start

### 1. Install Dependencies
```bash
cd train_with_huggingface
python install_huggingface.py
```

### 2. Test the Setup
```bash
python test_hf_integration.py
```

### 3. Start Training
```bash
# Quick test (5 minutes)
python train_with_huggingface.py --max-samples 100 --epochs 5

# Standard training (1-2 hours)
python train_with_huggingface.py --max-samples 2000 --epochs 50

# Full training (4-8 hours)
python train_with_huggingface.py --hf-dataset shapenet --max-samples 10000 --epochs 100
```

## Available Datasets

- **ModelNet** (recommended for first run) - Small, fast, good for testing
- **ShapeNet** - Large-scale, comprehensive 3D shapes
- **PartNet** - Fine-grained part segmentation
- **ScanNet** - Indoor scene understanding

## Key Features

- **Automatic dataset caching** - No need to re-download
- **Smart data preprocessing** - Handles various 3D formats
- **Memory optimization** - Configurable sample limits
- **Progress tracking** - Real-time training progress
- **Advanced loss functions** - Vertex, smoothness, and symmetry losses
- **Comprehensive logging** - Detailed training logs and plots

## Usage Examples

### Basic Training
```bash
python train_with_huggingface.py
```

### Custom Configuration
```bash
python train_with_huggingface.py \
    --hf-dataset shapenet \
    --max-samples 5000 \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.0005
```

### Quick Testing
```bash
python train_with_huggingface.py \
    --hf-dataset modelnet \
    --max-samples 100 \
    --max-val-samples 20 \
    --epochs 5 \
    --batch-size 8
```

## Output Files

After training, you'll find:
- **Checkpoints**: `../checkpoints/checkpoint_epoch_X.pth`
- **Best Model**: `../checkpoints/best_model.pth`
- **Final Model**: `../final_models/hf_trained_model.pth`
- **Training Logs**: `../logs/hf_training.log`
- **Training Plots**: `../plots/training_results.png`
- **Cached Datasets**: `../hf_cache/`

## Troubleshooting

### Common Issues
1. **Import errors**: Run `python install_huggingface.py`
2. **Dataset download issues**: Check internet connection
3. **Memory issues**: Reduce `--max-samples` and `--batch-size`
4. **CUDA out of memory**: Reduce model complexity with `--attention-heads` and `--d-model`

### Getting Help
1. Check training logs in `../logs/`
2. Run integration tests: `python test_hf_integration.py`
3. Verify GPU availability and memory
4. Check library versions

## Next Steps

1. **Start with testing**: Run the test script
2. **Quick training**: Use ModelNet with limited samples
3. **Scale up**: Gradually increase dataset size and complexity
4. **Customize**: Modify the training pipeline for your needs
5. **Evaluate**: Use the trained model with your prediction script

## Documentation

- **README_HuggingFace.md**: Comprehensive usage guide with examples
- **Code comments**: Detailed inline documentation
- **Training logs**: Real-time feedback during training

**Note**: This integration is designed to work alongside your existing training pipeline. It provides access to professional-grade datasets while maintaining compatibility with your current codebase.
