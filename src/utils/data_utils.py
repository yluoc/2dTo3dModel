import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from .common_utils import load_obj, create_directories

class ShapeDataset(Dataset):
    """
    Custom dataset for 2D images and 3D model vertices.
    """
    def __init__(self, img_path, mdl_path, obj_num, transform=None):
        """
        Initialize the dataset.
        
        Args:
            img_path (str): Path to 2D images directory
            mdl_path (str): Path to 3D models directory
            obj_num (int): Number of objects to load
            transform: Image transformations
        """
        import sys
        # Add parent directory to Python path for imports
        from ..dataPreprocess import dataPreprocess
        
        self.data_preprocess = dataPreprocess(img_path, mdl_path, obj_num)
        self.transform = transform
        self.mapped_dataset = self.data_preprocess.shape_model_match()
        
    def __len__(self):
        return len(self.mapped_dataset)
    
    def __getitem__(self, idx):
        img_filepath, mdl_filepath, _, _ = self.mapped_dataset[idx]
        
        # Load and preprocess image
        img = Image.open(img_filepath).convert('RGB')
        img = img.resize((128, 128))
        if self.transform:
            img = self.transform(img)
        
        # Load 3D model vertices
        vertices = load_obj(mdl_filepath)
        
        return img, vertices

def get_data_transforms():
    """
    Get standard data transformations for training.
    
    Returns:
        transforms.Compose: Image transformations
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_dataloader(img_path, mdl_path, obj_num, batch_size=32, shuffle=True):
    """
    Create a DataLoader for the shape dataset.
    
    Args:
        img_path (str): Path to 2D images directory
        mdl_path (str): Path to 3D models directory
        obj_num (int): Number of objects to load
        batch_size (int): Batch size for training
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    transform = get_data_transforms()
    dataset = ShapeDataset(img_path, mdl_path, obj_num, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader, dataset

def prepare_batch_data(batch, device):
    """
    Prepare batch data for training.
    
    Args:
        batch: Batch of data from DataLoader
        device: Device to place tensors on
        
    Returns:
        tuple: (images, targets) tensors
    """
    images, vertices_list = batch
    
    # Move images to device
    images = images.to(device)
    
    # Pad vertices to same length and move to device
    max_vertices = max(len(v) for v in vertices_list)
    vertices_padded = torch.zeros(len(vertices_list), max_vertices, 3, dtype=torch.float32)
    
    for i, vertices in enumerate(vertices_list):
        vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
        vertices_padded[i, :len(vertices), :] = vertices_tensor
    
    # Flatten vertices for model output
    vertices_flat = vertices_padded.view(vertices_padded.size(0), -1).to(device)
    
    return images, vertices_flat

def calculate_output_vertices(dataset):
    """
    Calculate the number of output vertices from the dataset.
    
    Args:
        dataset: ShapeDataset instance
        
    Returns:
        int: Number of output vertices
    """
    sample_vertices = dataset[0][1]
    return len(sample_vertices)

def get_dataset_info(img_path, mdl_path, obj_num):
    """
    Get information about the dataset.
    
    Args:
        img_path (str): Path to 2D images directory
        mdl_path (str): Path to 3D models directory
        obj_num (int): Number of objects to load
        
    Returns:
        dict: Dataset information
    """
    try:
        dataloader, dataset = create_dataloader(img_path, mdl_path, obj_num, batch_size=1)
        output_vertices = calculate_output_vertices(dataset)
        
        return {
            'dataset_size': len(dataset),
            'output_vertices': output_vertices,
            'image_size': (128, 128),
            'channels': 3
        }
    except Exception as e:
        return {
            'error': str(e),
            'dataset_size': 0,
            'output_vertices': 0,
            'image_size': (128, 128),
            'channels': 3
        }
