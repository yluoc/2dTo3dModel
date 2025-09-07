"""
Dataset utilities for 2D images and 3D models training.

This module provides dataset classes and utilities for training deep learning models
on paired 2D images and 3D models.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import trimesh
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class Image3DDataset(Dataset):
    """
    Dataset class for paired 2D images and 3D models.
    
    This dataset loads 2D images from the generated frame capture and their
    corresponding 3D models, creating pairs for training.
    """
    
    def __init__(self, 
                 img_dir: str = "./training_dataset/2dImg",
                 model_dir: str = "./training_dataset/3dModel", 
                 transform=None,
                 max_vertices: int = 10000,
                 normalize_models: bool = True):
        """
        Initialize the dataset.
        
        Args:
            img_dir: Directory containing 2D images organized by model
            model_dir: Directory containing 3D models
            transform: Image transformations to apply
            max_vertices: Maximum number of vertices to use (for memory efficiency)
            normalize_models: Whether to normalize 3D models to unit cube
        """
        self.img_dir = Path(img_dir)
        self.model_dir = Path(model_dir)
        self.transform = transform
        self.max_vertices = max_vertices
        self.normalize_models = normalize_models
        
        # Find all model directories
        self.model_dirs = [d for d in self.model_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(self.model_dirs)} model directories")
        
        # Create image-model pairs
        self.pairs = self._create_pairs()
        logger.info(f"Created {len(self.pairs)} image-model pairs")
        
    def _create_pairs(self) -> List[Dict[str, str]]:
        """
        Create pairs of 2D images and their corresponding 3D models.
        
        Returns:
            List of dictionaries containing image and model paths
        """
        pairs = []
        
        for model_dir in self.model_dirs:
            model_name = model_dir.name
            
            # Find corresponding image directory
            img_model_dir = self.img_dir / model_name
            
            if not img_model_dir.exists():
                logger.warning(f"No images found for model: {model_name}")
                continue
            
            # Find .obj files in model directory
            obj_files = list(model_dir.glob("*.obj"))
            if not obj_files:
                logger.warning(f"No .obj files found in model directory: {model_name}")
                continue
            
            # Use the first .obj file found
            model_file = obj_files[0]
            
            # Find all image files for this model
            image_files = list(img_model_dir.glob("*.png"))
            
            if not image_files:
                logger.warning(f"No images found for model: {model_name}")
                continue
            
            # Create pairs for each image
            for img_file in image_files:
                pairs.append({
                    'image_path': str(img_file),
                    'model_path': str(model_file),
                    'model_name': model_name,
                    'image_name': img_file.stem
                })
        
        return pairs
    
    def __len__(self) -> int:
        """Return the number of image-model pairs."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single image-model pair.
        
        Args:
            idx: Index of the pair to retrieve
            
        Returns:
            Tuple of (image_tensor, vertices_tensor)
        """
        pair = self.pairs[idx]
        
        # Load and preprocess image
        image = self._load_image(pair['image_path'])
        
        # Load and preprocess 3D model
        vertices = self._load_model(pair['model_path'])
        
        return image, vertices
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transformations if provided
            if self.transform:
                image = self.transform(image)
            else:
                # Default transformations
                image = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            # Return a dummy image
            return torch.zeros(3, 512, 512)
    
    def _load_model(self, model_path: str) -> torch.Tensor:
        """
        Load and preprocess a 3D model.
        
        Args:
            model_path: Path to the 3D model file
            
        Returns:
            Preprocessed vertices tensor
        """
        try:
            # Load 3D model
            loaded = trimesh.load(model_path)
            
            # Handle different return types from trimesh.load()
            if isinstance(loaded, trimesh.Scene):
                if len(loaded.geometry) > 0:
                    mesh = list(loaded.geometry.values())[0]
                    if not isinstance(mesh, trimesh.Trimesh):
                        raise ValueError("Scene contains non-mesh geometry")
                else:
                    raise ValueError("Scene is empty")
            elif isinstance(loaded, trimesh.Trimesh):
                mesh = loaded
            else:
                raise ValueError(f"Unknown geometry type: {type(loaded)}")
            
            # Extract vertices
            vertices = mesh.vertices.astype(np.float32)
            
            # Normalize if requested
            if self.normalize_models:
                vertices = self._normalize_vertices(vertices)
            
            # Limit number of vertices for memory efficiency
            if len(vertices) > self.max_vertices:
                # Randomly sample vertices
                indices = np.random.choice(len(vertices), self.max_vertices, replace=False)
                vertices = vertices[indices]
            
            # Convert to tensor
            vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
            
            return vertices_tensor
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            # Return dummy vertices
            return torch.zeros(self.max_vertices, 3)
    
    def _normalize_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """
        Normalize vertices to fit in a unit cube centered at origin.
        
        Args:
            vertices: Input vertices array
            
        Returns:
            Normalized vertices array
        """
        # Center the vertices
        centroid = np.mean(vertices, axis=0)
        vertices = vertices - centroid
        
        # Scale to fit in unit cube
        max_extent = np.max(np.abs(vertices))
        if max_extent > 0:
            vertices = vertices / max_extent
        
        return vertices


def get_data_transforms_2d3d(augment: bool = True) -> transforms.Compose:
    """
    Get data transformations for 2D-3D training.
    
    Args:
        augment: Whether to include data augmentation
        
    Returns:
        Composed transformations
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_2d3d_dataloader(img_dir: str = "./training_dataset/2dImg",
                          model_dir: str = "./training_dataset/3dModel",
                          batch_size: int = 8,
                          shuffle: bool = True,
                          num_workers: int = 0,
                          max_vertices: int = 10000,
                          augment: bool = True) -> Tuple[DataLoader, Image3DDataset]:
    """
    Create a DataLoader for 2D-3D training.
    
    Args:
        img_dir: Directory containing 2D images
        model_dir: Directory containing 3D models
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        max_vertices: Maximum number of vertices per model
        augment: Whether to use data augmentation
        
    Returns:
        Tuple of (DataLoader, Dataset)
    """
    transform = get_data_transforms_2d3d(augment=augment)
    
    dataset = Image3DDataset(
        img_dir=img_dir,
        model_dir=model_dir,
        transform=transform,
        max_vertices=max_vertices,
        normalize_models=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_2d3d
    )
    
    return dataloader, dataset


def collate_fn_2d3d(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for 2D-3D dataset.
    
    Args:
        batch: List of (image, vertices) tuples
        
    Returns:
        Tuple of batched (images, vertices)
    """
    images, vertices_list = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Use fixed size for all vertices (max_vertices from dataset)
    max_vertices = 5000  # Fixed maximum vertices
    vertices_padded = torch.zeros(len(vertices_list), max_vertices, 3, dtype=torch.float32)
    
    for i, vertices in enumerate(vertices_list):
        # Truncate or pad to max_vertices
        actual_vertices = min(vertices.size(0), max_vertices)
        vertices_padded[i, :actual_vertices, :] = vertices[:actual_vertices]
    
    # Flatten vertices for model output
    vertices_flat = vertices_padded.view(vertices_padded.size(0), -1)
    
    return images, vertices_flat


def get_dataset_info_2d3d(img_dir: str = "./training_dataset/2dImg",
                         model_dir: str = "./training_dataset/3dModel") -> Dict[str, Any]:
    """
    Get information about the 2D-3D dataset.
    
    Args:
        img_dir: Directory containing 2D images
        model_dir: Directory containing 3D models
        
    Returns:
        Dictionary containing dataset information
    """
    try:
        dataset = Image3DDataset(img_dir=img_dir, model_dir=model_dir)
        
        # Get sample data to determine dimensions
        sample_image, sample_vertices = dataset[0]
        
        # Use fixed output dimension for consistency
        fixed_max_vertices = 5000
        fixed_output_dim = fixed_max_vertices * 3
        
        return {
            'dataset_size': len(dataset),
            'num_models': len(dataset.model_dirs),
            'image_shape': sample_image.shape,
            'max_vertices': fixed_max_vertices,
            'vertices_shape': (fixed_max_vertices, 3),
            'output_dimension': fixed_output_dim
        }
        
    except Exception as e:
        logger.error(f"Failed to get dataset info: {str(e)}")
        return {
            'error': str(e),
            'dataset_size': 0,
            'num_models': 0,
            'image_shape': (3, 512, 512),
            'max_vertices': 10000,
            'vertices_shape': (10000, 3),
            'output_dimension': 30000
        }


def split_dataset(dataset: Image3DDataset, train_ratio: float = 0.8) -> Tuple[Image3DDataset, Image3DDataset]:
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: Original dataset
        train_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    return train_dataset, val_dataset
