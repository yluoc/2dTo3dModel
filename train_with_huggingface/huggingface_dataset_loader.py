"""
Hugging Face Dataset Loader for 2D to 3D Model Training

This module provides functionality to load and preprocess datasets from Hugging Face
for training 2D to 3D reconstruction models. It supports various dataset formats
and automatically handles data preprocessing and augmentation.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
from pathlib import Path
import zipfile
import tempfile
from tqdm import tqdm
import logging

# Try to import Hugging Face libraries
try:
    from datasets import load_dataset, Dataset, Features, Value, Image as HFImage
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: Hugging Face libraries not available. Install with: pip install datasets huggingface-hub")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFace2D3DDataset(Dataset):
    """
    Dataset class for loading 2D to 3D data from Hugging Face datasets.
    """
    
    def __init__(self, dataset_name, split="train", transform=None, max_samples=None, 
                 cache_dir="./hf_cache", download_mode="reuse_cache_if_exists"):
        """
        Initialize the Hugging Face dataset loader.
        
        Args:
            dataset_name (str): Name of the Hugging Face dataset
            split (str): Dataset split (train, validation, test)
            transform: Image transformations
            max_samples (int): Maximum number of samples to load
            cache_dir (str): Directory to cache downloaded datasets
            download_mode (str): Download mode for datasets
        """
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform or self._get_default_transforms()
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        self.download_mode = download_mode
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Validate dataset structure
        self._validate_dataset()
        
        logger.info(f"Loaded {len(self.dataset)} samples from {dataset_name}")
    
    def _load_dataset(self):
        """Load dataset from Hugging Face."""
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face libraries not available")
        
        try:
            # Load dataset with caching
            dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
                download_mode=self.download_mode
            )
            
            # Limit samples if specified
            if self.max_samples and len(dataset) > self.max_samples:
                dataset = dataset.select(range(self.max_samples))
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise
    
    def _validate_dataset(self):
        """Validate that the dataset has the required structure."""
        if len(self.dataset) == 0:
            raise ValueError(f"Dataset {self.dataset_name} is empty")
        
        # Check if dataset has required columns
        required_columns = ['image', 'vertices', 'faces']
        available_columns = self.dataset.column_names
        
        # Check for common variations
        if 'image' not in available_columns:
            # Look for image-related columns
            image_columns = [col for col in available_columns if 'image' in col.lower() or 'img' in col.lower()]
            if image_columns:
                logger.warning(f"Dataset uses column '{image_columns[0]}' instead of 'image'")
            else:
                raise ValueError(f"Dataset must contain image data. Available columns: {available_columns}")
        
        if 'vertices' not in available_columns:
            # Look for 3D data columns
            mesh_columns = [col for col in available_columns if 'mesh' in col.lower() or '3d' in col.lower() or 'obj' in col.lower()]
            if mesh_columns:
                logger.warning(f"Dataset uses column '{mesh_columns[0]}' instead of 'vertices'")
            else:
                logger.warning("Dataset may not contain 3D vertex data")
    
    def _get_default_transforms(self):
        """Get default image transformations."""
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        try:
            sample = self.dataset[idx]
            
            # Extract image
            if 'image' in sample:
                image = sample['image']
            else:
                # Look for alternative image columns
                image_cols = [col for col in sample.keys() if 'image' in col.lower() or 'img' in col.lower()]
                if image_cols:
                    image = sample[image_cols[0]]
                else:
                    raise ValueError(f"No image data found in sample. Available keys: {list(sample.keys())}")
            
            # Convert PIL Image if needed
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                elif isinstance(image, str):
                    # Load image from path
                    image = Image.open(image).convert('RGB')
                else:
                    raise ValueError(f"Unsupported image format: {type(image)}")
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            # Extract 3D data
            vertices = self._extract_vertices(sample)
            faces = self._extract_faces(sample)
            
            return {
                'image': image,
                'vertices': vertices,
                'faces': faces,
                'metadata': {k: v for k, v in sample.items() 
                           if k not in ['image', 'vertices', 'faces']}
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a default sample or re-raise
            raise
    
    def _extract_vertices(self, sample):
        """Extract vertex data from sample."""
        if 'vertices' in sample:
            vertices = sample['vertices']
        elif 'mesh' in sample:
            vertices = sample['mesh'].get('vertices', [])
        elif '3d_data' in sample:
            vertices = sample['3d_data'].get('vertices', [])
        else:
            # Create dummy vertices if none available
            logger.warning("No vertex data found, creating dummy vertices")
            vertices = np.random.rand(100, 3).astype(np.float32)
        
        # Convert to numpy array if needed
        if isinstance(vertices, list):
            vertices = np.array(vertices, dtype=np.float32)
        elif isinstance(vertices, torch.Tensor):
            vertices = vertices.numpy()
        
        return vertices
    
    def _extract_faces(self, sample):
        """Extract face data from sample."""
        if 'faces' in sample:
            faces = sample['faces']
        elif 'mesh' in sample:
            faces = sample['mesh'].get('faces', [])
        elif '3d_data' in sample:
            faces = sample['3d_data'].get('faces', [])
        else:
            # Create dummy faces if none available
            logger.warning("No face data found, creating dummy faces")
            vertices = self._extract_vertices(sample)
            # Simple triangulation
            faces = np.array([[0, 1, 2] + [i % (len(vertices)-2) + 3 for i in range(len(vertices)-3)]], dtype=np.int32)
        
        # Convert to numpy array if needed
        if isinstance(faces, list):
            faces = np.array(faces, dtype=np.int32)
        elif isinstance(faces, torch.Tensor):
            faces = faces.numpy()
        
        return faces


class DatasetManager:
    """
    Manager class for handling multiple Hugging Face datasets and data preparation.
    """
    
    def __init__(self, cache_dir="./hf_cache"):
        """
        Initialize the dataset manager.
        
        Args:
            cache_dir (str): Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.available_datasets = self._get_available_datasets()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_available_datasets(self):
        """Get list of available 2D to 3D datasets."""
        return {
            "shapenet": {
                "name": "shapenet/ShapeNetCore",
                "description": "ShapeNet Core dataset with 3D models and renderings",
                "features": ["3D models", "2D renderings", "Multiple views"],
                "size": "Large",
                "license": "Academic"
            },
            "modelnet": {
                "name": "modelnet/ModelNet40",
                "description": "3D CAD models with multiple views",
                "features": ["CAD models", "Multiple views", "40 categories"],
                "size": "Medium",
                "license": "Academic"
            },
            "partnet": {
                "name": "partnet/PartNet",
                "description": "Fine-grained part segmentation dataset",
                "features": ["Part segmentation", "Hierarchical structure", "Multiple objects"],
                "size": "Large",
                "license": "Academic"
            },
            "scannet": {
                "name": "scannet/ScanNet",
                "description": "Indoor scene understanding dataset",
                "features": ["Indoor scenes", "3D scans", "Semantic labels"],
                "size": "Large",
                "license": "Academic"
            }
        }
    
    def list_datasets(self):
        """List all available datasets with descriptions."""
        print("Available 2D to 3D Datasets:")
        print("=" * 50)
        
        for key, info in self.available_datasets.items():
            print(f"\n{key.upper()}:")
            print(f"  Description: {info['description']}")
            print(f"  Features: {', '.join(info['features'])}")
            print(f"  Size: {info['size']}")
            print(f"  License: {info['license']}")
    
    def load_dataset(self, dataset_key, split="train", max_samples=None, **kwargs):
        """
        Load a specific dataset.
        
        Args:
            dataset_key (str): Key of the dataset to load
            split (str): Dataset split
            max_samples (int): Maximum number of samples
            **kwargs: Additional arguments for dataset loading
            
        Returns:
            HuggingFace2D3DDataset: Loaded dataset
        """
        if dataset_key not in self.available_datasets:
            raise ValueError(f"Unknown dataset: {dataset_key}. Use list_datasets() to see available options.")
        
        dataset_name = self.available_datasets[dataset_key]["name"]
        
        try:
            dataset = HuggingFace2D3DDataset(
                dataset_name=dataset_name,
                split=split,
                max_samples=max_samples,
                cache_dir=self.cache_dir,
                **kwargs
            )
            
            logger.info(f"Successfully loaded {dataset_key} dataset")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_key}: {e}")
            raise
    
    def create_dataloader(self, dataset, batch_size=32, shuffle=True, num_workers=0, **kwargs):
        """
        Create a PyTorch DataLoader from the dataset.
        
        Args:
            dataset: HuggingFace2D3DDataset instance
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle data
            num_workers (int): Number of worker processes
            **kwargs: Additional DataLoader arguments
            
        Returns:
            DataLoader: PyTorch DataLoader
        """
        def collate_fn(batch):
            """Custom collate function for the dataset."""
            images = torch.stack([item['image'] for item in batch])
            
            # Handle variable-length vertices and faces
            vertices_list = [item['vertices'] for item in batch]
            faces_list = [item['faces'] for item in batch]
            
            # Pad vertices to same length
            max_vertices = max(len(v) for v in vertices_list)
            padded_vertices = []
            
            for vertices in vertices_list:
                if len(vertices) < max_vertices:
                    # Pad with zeros
                    padding = np.zeros((max_vertices - len(vertices), 3), dtype=np.float32)
                    padded = np.vstack([vertices, padding])
                else:
                    padded = vertices[:max_vertices]
                padded_vertices.append(padded)
            
            vertices_tensor = torch.tensor(np.array(padded_vertices), dtype=torch.float32)
            
            # Handle faces similarly
            max_faces = max(len(f) for f in faces_list)
            padded_faces = []
            
            for faces in faces_list:
                if len(faces) < max_faces:
                    # Pad with zeros
                    padding = np.zeros((max_faces - len(faces), 3), dtype=np.int32)
                    padded = np.vstack([faces, padding])
                else:
                    padded = faces[:max_faces]
                padded_faces.append(padded)
            
            faces_tensor = torch.tensor(np.array(padded_faces), dtype=torch.int32)
            
            return {
                'images': images,
                'vertices': vertices_tensor,
                'faces': faces_tensor,
                'metadata': [item['metadata'] for item in batch]
            }
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **kwargs
        )
    
    def download_custom_dataset(self, dataset_url, dataset_name, target_dir=None):
        """
        Download a custom dataset from a URL.
        
        Args:
            dataset_url (str): URL to download the dataset from
            dataset_name (str): Name for the dataset
            target_dir (str): Target directory for download
            
        Returns:
            str: Path to downloaded dataset
        """
        if target_dir is None:
            target_dir = os.path.join(self.cache_dir, dataset_name)
        
        os.makedirs(target_dir, exist_ok=True)
        
        # Download dataset
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        # Extract dataset
        try:
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            logger.info(f"Dataset downloaded and extracted to {target_dir}")
            return target_dir
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)


def create_hf_dataloader(dataset_key="shapenet", split="train", batch_size=32, 
                         max_samples=None, cache_dir="./hf_cache", **kwargs):
    """
    Convenience function to create a Hugging Face dataloader.
    
    Args:
        dataset_key (str): Key of the dataset to load
        split (str): Dataset split
        batch_size (int): Batch size for training
        max_samples (int): Maximum number of samples to load
        cache_dir (str): Directory to cache datasets
        **kwargs: Additional arguments
        
    Returns:
        tuple: (DataLoader, Dataset) for training
    """
    manager = DatasetManager(cache_dir=cache_dir)
    
    # Load dataset
    dataset = manager.load_dataset(dataset_key, split=split, max_samples=max_samples, **kwargs)
    
    # Create dataloader
    dataloader = manager.create_dataloader(dataset, batch_size=batch_size, **kwargs)
    
    return dataloader, dataset


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset loader
    try:
        # List available datasets
        manager = DatasetManager()
        manager.list_datasets()
        
        # Try to load a small dataset for testing
        print("\nTesting dataset loading...")
        test_dataloader, test_dataset = create_hf_dataloader(
            dataset_key="modelnet",  # Use ModelNet as it's smaller
            split="train",
            max_samples=10,  # Just 10 samples for testing
            batch_size=2
        )
        
        print(f"Test dataset loaded successfully with {len(test_dataset)} samples")
        
        # Test a batch
        for batch in test_dataloader:
            print(f"Batch keys: {batch.keys()}")
            print(f"Images shape: {batch['images'].shape}")
            print(f"Vertices shape: {batch['vertices'].shape}")
            print(f"Faces shape: {batch['faces'].shape}")
            break
            
    except Exception as e:
        print(f"Error during testing: {e}")
        print("This is normal if Hugging Face libraries are not installed or datasets are not accessible.")
