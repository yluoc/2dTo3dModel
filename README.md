# CNN-Based 2D Image to 3D Model Reconstruction

This project implements a deep learning approach to reconstruct 3D models from 2D images using Convolutional Neural Networks (CNNs) with PyTorch. The system captures multiple viewpoints of 3D models and trains a neural network to learn the mapping from 2D images to 3D geometry.

## Project Structure

```
2dTo3dModel/
├── README.md                           # Main project documentation
├── requirements.txt                    # Core project dependencies
├── train_2d3d_model.py                 # Main training script for 2D-3D reconstruction
├── capture_all_frames.py              # Script to generate 2D images from 3D models
├── monitor_training.py                 # Training progress monitoring tool
├── src/                               # Source code directory
│   ├── __init__.py                    # Package initialization
│   ├── cnnModel_pytorch.py           # Enhanced CNN model with attention mechanisms
│   ├── model_prediction.py           # Model prediction and 3D generation
│   └── utils/                        # Utility modules
│       ├── __init__.py               # Package initialization
│       ├── common_utils.py          # Common utility functions
│       ├── dataset_2d3d.py          # 2D-3D dataset handling
│       ├── frame_capture.py         # 3D model frame capture utilities
│       └── model_utils.py           # Model-related utilities
├── training_dataset/                  # Training data
│   ├── 2dImg/                        # Generated 2D images (900+ images)
│   │   ├── Alien_Animal/            # Images from multiple angles
│   │   ├── Apple_Rack/              # 36 images per model
│   │   └── ...                      # 26 different models
│   └── 3dModel/                     # Original 3D models (.obj files)
│       ├── Alien_Animal/            # 3D model files
│       ├── Apple_Rack/              # Textures and materials
│       └── ...                      # 26 different models
├── checkpoints/                      # Training checkpoints (saved every 10 epochs)
├── final_models/                     # Final trained models
└── logs/                             # Training logs and visualizations
```

## Features

- **Multi-viewpoint 2D image generation** from 3D models (36 angles per model)
- **Enhanced CNN architecture** with attention mechanisms and residual blocks
- **Advanced loss functions** including vertex, smoothness, symmetry, and chamfer losses
- **GPU acceleration** support with CUDA optimization
- **Comprehensive training pipeline** with early stopping and learning rate scheduling
- **Real-time training monitoring** with detailed progress tracking
- **Modular architecture** for easy customization and extension

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Training Data (if not already done)
```bash
python capture_all_frames.py
```

### 3. Train the Model
```bash
python train_2d3d_model.py
```

### 4. Monitor Training Progress
```bash
python monitor_training.py
```

### 5. Generate 3D Models from New Images
```bash
python src/model_prediction.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- See `requirements.txt` for full dependencies

## Usage

### Dataset Generation
```bash
# Generate 2D images from 3D models
python capture_all_frames.py
```

### Training Configuration
Modify training parameters in `train_2d3d_model.py`:
```python
config = {
    'img_dir': '2dImg path',             # input 2d image path
    'model_dir': '3dMdl path',             # input 3d model path
    'batch_size': 4,           # Adjust based on GPU memory
    'learning_rate': 0.0001,    # Learning rate
    'max_vertices': 5000,       # Maximum vertices per model
    'attention_heads': 8,       # Number of attention heads
    'd_model': 512,             # Model dimension
    'epochs': 100             # Number of training epochs  
}
epochs = config['epochs']
```

### Training
```bash
# Start training with current configuration
python train_2d3d_model.py
```

### Monitoring
```bash
# Monitor training progress
python monitor_training.py
```

### Prediction
```bash
# Generate 3D models from 2D images
python src/model_prediction.py
```

## Dataset Information

- **Total Models**: 26 different 3D models
- **Images per Model**: 36 images (12 azimuth × 3 elevation angles)
- **Total Images**: 900+ 2D images
- **Image Resolution**: 512×512 pixels
- **3D Model Format**: .obj files with textures and materials
- **Vertex Count**: Up to 5,000 vertices per model (configurable)

## Model Architecture

- **Base Architecture**: Enhanced CNN with residual blocks
- **Attention Mechanisms**: CBAM (Channel and Spatial Attention)
- **Feature Extraction**: Feature Pyramid Network for multi-scale features
- **Transformer Components**: Multi-head attention for feature refinement
- **Model Parameters**: ~15M parameters
- **Input**: 512×512 RGB images
- **Output**: Flattened 3D vertex coordinates (15,000 dimensions)

## Training Features

- **Advanced Loss Functions**:
  - Vertex Loss: MSE between predicted and target vertices
  - Smoothness Loss: Encourages smooth 3D surfaces
  - Symmetry Loss: Promotes symmetric shapes
  - Chamfer Loss: Better 3D reconstruction quality

- **Training Optimizations**:
  - Early stopping with patience-based stopping
  - Learning rate scheduling with ReduceLROnPlateau
  - Gradient clipping for training stability
  - Comprehensive logging and visualization

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
