import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

def load_obj(filename):
    """
    Load vertices from an OBJ file.
    
    Args:
        filename (str): Path to the OBJ file
        
    Returns:
        np.ndarray: Array of vertices with shape (N, 3)
    """
    vertices = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices, dtype=np.float32)

def save_obj(vertices, filename):
    """
    Save vertices to an OBJ file.
    
    Args:
        vertices (np.ndarray): Array of vertices with shape (N, 3)
        filename (str): Path to save the OBJ file
    """
    with open(filename, 'w') as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess an image for model input.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size (width, height)
        
    Returns:
        PIL.Image: Preprocessed image
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    return image

def tensor_to_numpy(tensor):
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        tensor (torch.Tensor): Input tensor
        
    Returns:
        np.ndarray: Numpy array
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.detach().numpy()

def numpy_to_tensor(array, device='cuda'):
    """
    Convert numpy array to PyTorch tensor.
    
    Args:
        array (np.ndarray): Input numpy array
        device (str): Device to place tensor on
        
    Returns:
        torch.Tensor: PyTorch tensor
    """
    return torch.tensor(array, dtype=torch.float32).to(device)

def create_directories(*dirs):
    """
    Create multiple directories if they don't exist.
    
    Args:
        *dirs: Variable number of directory paths
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def plot_training_curves(train_losses, val_losses, save_path='./training_curves.png'):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def calculate_vertex_count(obj_file_path):
    """
    Calculate the number of vertices in an OBJ file.
    
    Args:
        obj_file_path (str): Path to the OBJ file
        
    Returns:
        int: Number of vertices
    """
    vertices = load_obj(obj_file_path)
    return len(vertices)

def get_device_info():
    """
    Get information about available devices.
    
    Returns:
        dict: Device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if torch.cuda.is_available():
        device_info.update({
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'gpu_count': torch.cuda.device_count()
        })
    
    return device_info
