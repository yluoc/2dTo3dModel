# CNN-Based 2D Image to 3D Model Reconstruction

This project implements a deep learning approach to reconstruct 3D models from 2D images using Convolutional Neural Networks (CNNs) with PyTorch.

## Project Structure

```
2dTo3dModel/
├── README.md                           # Main project documentation
├── requirements.txt                    # Core project dependencies
├── cnnModel_pytorch.py               # CNN model architecture
├── train_model_pytorch_gpu.py        # Main training script
├── model_prediction.py               # Model prediction and 3D generation
├── dataPreprocess.py                 # Data preprocessing utilities
├── utils/                            # Utility modules
│   ├── common_utils.py              # Common utility functions
│   ├── data_utils.py                # Data handling utilities
│   └── model_utils.py               # Model-related utilities
├── train_with_huggingface/           # Hugging Face dataset integration
│   ├── README.md                     # Hugging Face usage guide
│   ├── requirements_hf.txt           # Hugging Face dependencies
│   ├── huggingface_dataset_loader.py # Dataset loading functionality
│   ├── train_with_huggingface.py    # Training with HF datasets
│   ├── test_hf_integration.py       # Integration testing
│   └── install_huggingface.py       # Automated setup
├── checkpoints/                      # Training checkpoints
├── final_models/                     # Trained models
├── shapes2d/                         # 2D image dataset
└── shapes3d/                         # 3D model dataset
```

## Features

- **CNN-based 3D reconstruction** from 2D images
- **GPU acceleration** support with CUDA
- **Attention mechanisms** for enhanced performance
- **Advanced loss functions** including smoothness and symmetry
- **Hugging Face integration** for large-scale datasets
- **Comprehensive logging** and visualization
- **Modular architecture** for easy customization

## Quick Start

### 1. Install Core Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train with Local Dataset
```bash
python train_model_pytorch_gpu.py
```

### 3. Use Hugging Face Datasets (Optional)
```bash
cd train_with_huggingface
python install_huggingface.py
python train_with_huggingface.py
```

### 4. Generate 3D Models
```bash
python model_prediction.py
```

## Hugging Face Integration

For training with large-scale, professional datasets, see the `train_with_huggingface/` subdirectory. This provides access to:

- **ShapeNet**: Large-scale 3D shape dataset
- **ModelNet**: 3D CAD models with 40 categories
- **PartNet**: Fine-grained part segmentation
- **ScanNet**: Indoor scene understanding

See `train_with_huggingface/README.md` for detailed usage instructions.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- See `requirements.txt` for full dependencies

## Usage

### Training
```bash
# Local dataset training
python train_model_pytorch_gpu.py

# Hugging Face dataset training
cd train_with_huggingface
python train_with_huggingface.py --hf-dataset modelnet --max-samples 1000
```

### Prediction
```bash
python model_prediction.py
```

## Documentation

- **Main README**: This file - project overview and basic usage
- **Hugging Face README**: `train_with_huggingface/README.md` - detailed HF usage
- **Code comments**: Comprehensive inline documentation
- **Training logs**: Real-time feedback during training

## License

This project follows the MIT license. Hugging Face datasets are subject to their respective licenses (typically academic use).

## Contributing

Contributions are welcome! Please see the individual README files for specific contribution guidelines.
