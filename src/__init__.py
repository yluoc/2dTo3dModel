"""
CNN-Based 2D Image to 3D Model Reconstruction

This package contains the core functionality for reconstructing 3D models
from 2D images using Convolutional Neural Networks with PyTorch.
"""

__version__ = "1.0.0"
__author__ = "2D to 3D Model Team"

# Import main modules for easy access
from .cnnModel_pytorch import EnhancedCNNModel
from .dataPreprocess import dataPreprocess
from .model_prediction import EnhancedPredModel
from .train_model_pytorch_gpu import EnhancedTrainModel

__all__ = [
    "EnhancedCNNModel",
    "dataPreprocess", 
    "EnhancedPredModel",
    "EnhancedTrainModel"
]
