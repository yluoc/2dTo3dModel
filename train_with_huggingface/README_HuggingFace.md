# Hugging Face Dataset Integration for 2D to 3D Model Training

This document explains how to use the Hugging Face dataset integration to train your 2D to 3D reconstruction model with large-scale, high-quality datasets.

## Overview

The Hugging Face integration provides access to several high-quality 2D to 3D datasets:
- **ShapeNet**: Large-scale 3D shape dataset with multiple views
- **ModelNet**: 3D CAD models with 40 categories
- **PartNet**: Fine-grained part segmentation dataset
- **ScanNet**: Indoor scene understanding dataset

## Installation

### 1. Install Dependencies

First, install the required Hugging Face libraries:

```bash
pip install -r requirements.txt
```

This will install:
- `datasets>=2.14.0` - Hugging Face datasets library
- `huggingface-hub>=0.16.0` - Hugging Face hub integration
- `transformers>=4.30.0` - Additional utilities

### 2. Verify Installation

Run the test script to verify everything is working:

```bash
python test_hf_integration.py
```

This will test:
- Library imports
- Dataset loader functionality
- Small dataset loading
- Training script integration

## Quick Start

### 1. Basic Training with ModelNet Dataset

Start training with the default configuration using ModelNet dataset:

```bash
python train_with_huggingface.py
```

This will:
- Download and cache the ModelNet dataset
- Train for 50 epochs with 1000 samples
- Save checkpoints and final model
- Generate training plots

### 2. Customize Training Parameters

```bash
python train_with_huggingface.py \
    --hf-dataset shapenet \
    --max-samples 5000 \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.0005
```

### 3. Use Different Datasets

```bash
# ShapeNet (larger, more complex shapes)
python train_with_huggingface.py --hf-dataset shapenet --max-samples 2000

# PartNet (fine-grained parts)
python train_with_huggingface.py --hf-dataset partnet --max-samples 1500

# ScanNet (indoor scenes)
python train_with_huggingface.py --hf-dataset scannet --max-samples 1000
```

## Available Datasets

### 1. ShapeNet Core
- **Description**: Large-scale 3D shape dataset with multiple views
- **Size**: Very Large (~51GB)
- **Features**: 3D models, 2D renderings, multiple views
- **Best for**: General 3D reconstruction, complex shapes
- **License**: Academic

### 2. ModelNet40
- **Description**: 3D CAD models with multiple views
- **Size**: Medium (~1.6GB)
- **Features**: CAD models, 40 categories, multiple views
- **Best for**: Quick testing, CAD-like objects
- **License**: Academic

### 3. PartNet
- **Description**: Fine-grained part segmentation dataset
- **Size**: Large (~20GB)
- **Features**: Part segmentation, hierarchical structure
- **Best for**: Detailed part understanding
- **License**: Academic

### 4. ScanNet
- **Description**: Indoor scene understanding dataset
- **Size**: Large (~25GB)
- **Features**: Indoor scenes, 3D scans, semantic labels
- **Best for**: Indoor scene reconstruction
- **License**: Academic

## Configuration Options

### Dataset Configuration
```python
config = {
    'use_huggingface': True,
    'hf_dataset': 'modelnet',  # shapenet, modelnet, partnet, scannet
    'max_samples': 1000,       # Limit samples for faster training
    'max_val_samples': 200,    # Validation samples
    'hf_cache_dir': './hf_cache'  # Dataset cache directory
}
```

### Model Configuration
```python
config = {
    'img_height': 128,
    'img_width': 128,
    'channels': 3,
    'attention_heads': 8,
    'd_model': 512
}
```

### Training Configuration
```python
config = {
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 0.0001,
    'optimizer': 'adam',
    'early_stopping': True,
    'patience': 10
}
```

### Loss Weights
```python
config = {
    'vertex_weight': 1.0,        # Main vertex coordinate loss
    'smoothness_weight': 0.1,    # Surface smoothness
    'symmetry_weight': 0.05      # Shape symmetry
}
```

## Advanced Usage

### 1. Custom Dataset Integration

You can integrate your own datasets alongside Hugging Face datasets:

```python
from huggingface_dataset_loader import DatasetManager

manager = DatasetManager()

# Load Hugging Face dataset
hf_dataset = manager.load_dataset('modelnet', split='train', max_samples=1000)

# Load your local dataset
from utils import create_dataloader
local_dataloader, local_dataset = create_dataloader(
    './shapes2d', './shapes3d', 100, batch_size=16
)

# Combine datasets if needed
# (Implementation depends on your specific needs)
```

### 2. Custom Data Preprocessing

Extend the dataset loader for custom preprocessing:

```python
from huggingface_dataset_loader import HuggingFace2D3DDataset
from torchvision import transforms

class CustomDataset(HuggingFace2D3DDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Custom transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Larger images
            transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __getitem__(self, idx):
        # Custom data loading logic
        sample = super().__getitem__(idx)
        
        # Apply custom preprocessing
        # ... your custom logic here
        
        return sample
```

### 3. Multi-GPU Training

For multi-GPU training, modify the training script:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed training
dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model)
```

## Performance Tips

### 1. Dataset Caching
- Datasets are automatically cached in `./hf_cache`
- First run will download the dataset (may take time)
- Subsequent runs will use cached data

### 2. Memory Management
- Start with smaller datasets (ModelNet) for testing
- Use `max_samples` to limit dataset size
- Adjust batch size based on your GPU memory

### 3. Training Speed
- Use GPU acceleration when available
- Start with fewer epochs for quick testing
- Use early stopping to prevent overfitting

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Install missing dependencies
pip install datasets huggingface-hub transformers
```

#### 2. Dataset Download Issues
```bash
# Check internet connection
# Clear cache and retry
rm -rf ./hf_cache
python train_with_huggingface.py
```

#### 3. Memory Issues
```bash
# Reduce batch size and dataset size
python train_with_huggingface.py --batch-size 8 --max-samples 500
```

#### 4. CUDA Out of Memory
```bash
# Reduce model complexity
python train_with_huggingface.py --attention-heads 4 --d-model 256
```

### Getting Help

1. **Check logs**: Training logs are saved in `./logs/`
2. **Run tests**: Use `python test_hf_integration.py`
3. **Verify setup**: Check GPU availability and memory
4. **Check versions**: Ensure compatible library versions

## Example Training Sessions

### Quick Test (5 minutes)
```bash
python train_with_huggingface.py \
    --hf-dataset modelnet \
    --max-samples 100 \
    --max-val-samples 20 \
    --epochs 5 \
    --batch-size 8
```

### Standard Training (1-2 hours)
```bash
python train_with_huggingface.py \
    --hf-dataset modelnet \
    --max-samples 2000 \
    --max-val-samples 400 \
    --epochs 50 \
    --batch-size 16
```

### Full Training (4-8 hours)
```bash
python train_with_huggingface.py \
    --hf-dataset shapenet \
    --max-samples 10000 \
    --max-val-samples 2000 \
    --epochs 100 \
    --batch-size 32
```

## Output Files

After training, you'll find:

- **Checkpoints**: `./checkpoints/checkpoint_epoch_X.pth`
- **Best Model**: `./checkpoints/best_model.pth`
- **Final Model**: `./final_models/hf_trained_model.pth`
- **Training Logs**: `./logs/hf_training.log`
- **Training Plots**: `./plots/training_results.png`
- **Cached Datasets**: `./hf_cache/`

## Next Steps

1. **Start with testing**: Run `python test_hf_integration.py`
2. **Quick training**: Use ModelNet with limited samples
3. **Scale up**: Gradually increase dataset size and complexity
4. **Customize**: Modify the training pipeline for your specific needs
5. **Evaluate**: Use the trained model with your prediction script

## Contributing

To add support for new datasets:

1. Update `_get_available_datasets()` in `DatasetManager`
2. Add dataset-specific preprocessing in `HuggingFace2D3DDataset`
3. Test with the integration test suite
4. Update this documentation

## License

This integration follows the same license as the original project. The Hugging Face datasets are subject to their respective licenses (typically academic use).
