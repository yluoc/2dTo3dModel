# CNN-Based 2D Image to 3D Model Reconstruction

This project implements a deep learning approach to reconstruct 3D models from 2D images using Convolutional Neural Networks (CNNs) with PyTorch.

## Project Structure

```
2dTo3dModel/
├── README.md                           # Main project documentation
├── requirements.txt                    # Core project dependencies
├── src/                               # Source code directory
│   ├── cnnModel_pytorch.py           # CNN model architecture
│   ├── train_model_pytorch_gpu.py    # Main training script
│   ├── model_prediction.py           # Model prediction and 3D generation
│   ├── dataPreprocess.py             # Data preprocessing utilities
│   └── utils/                        # Utility modules
│       ├── __init__.py               # Package initialization
│       ├── common_utils.py          # Common utility functions
│       ├── data_utils.py            # Data handling utilities
│       └── model_utils.py           # Model-related utilities
├── checkpoints/                      # Training checkpoints
├── final_models/                     # Trained models
├── logs/                             # Training and execution logs
├── plots/                            # Generated plots and visualizations
└── __pycache__/                      # Python cache files
```

## Features

- **CNN-based 3D reconstruction** from 2D images
- **GPU acceleration** support with CUDA
- **Attention mechanisms** for enhanced performance
- **Advanced loss functions** including smoothness and symmetry
- **Comprehensive logging** and visualization
- **Modular architecture** for easy customization

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python src/train_model_pytorch_gpu.py
```

### 3. Generate 3D Models
```bash
python src/model_prediction.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- See `requirements.txt` for full dependencies

## Usage

### Training
```bash
# Train with local dataset
python src/train_model_pytorch_gpu.py
```

### Prediction
```bash
python src/model_prediction.py
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
