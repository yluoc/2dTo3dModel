import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .common_utils import create_directories

def create_optimizer(model, lr=0.001, optimizer_type='adam'):
    """
    Create an optimizer for the model.
    
    Args:
        model: PyTorch model
        lr (float): Learning rate
        optimizer_type (str): Type of optimizer ('adam', 'sgd', 'rmsprop')
        
    Returns:
        torch.optim.Optimizer: Optimizer instance
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def create_scheduler(optimizer, mode='min', factor=0.5, patience=5, verbose=True):
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        mode (str): Mode for scheduler ('min' or 'max')
        factor (float): Factor to multiply learning rate by
        patience (int): Number of epochs with no improvement
        verbose (bool): Whether to print scheduler messages
        
    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau: Scheduler instance
    """
    return ReduceLROnPlateau(
        optimizer, 
        mode=mode, 
        factor=factor, 
        patience=patience, 
        verbose=verbose
    )

def save_checkpoint(model, optimizer, scheduler, epoch, save_dir='./checkpoints'):
    """
    Save a model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: PyTorch scheduler
        epoch (int): Current epoch number
        save_dir (str): Directory to save checkpoint
    """
    create_directories(save_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    checkpoint_path = f'{save_dir}/pytorch_model_epoch_{epoch:02d}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load a model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: PyTorch scheduler
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        int: Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f'Checkpoint loaded from epoch {epoch}')
    
    return epoch

def save_model(model, save_path='./final_models/pytorch_model_final.pth'):
    """
    Save the final model.
    
    Args:
        model: PyTorch model
        save_path (str): Path to save the model
    """
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved as '{save_path}'")

def load_model(model, model_path):
    """
    Load a saved model.
    
    Args:
        model: PyTorch model instance
        model_path (str): Path to saved model
        
    Returns:
        PyTorch model: Model with loaded weights
    """
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from '{model_path}'")
    return model

def count_parameters(model):
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_info(model):
    """
    Get information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Model information
    """
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': model.__class__.__name__
    }

def set_model_mode(model, training=True):
    """
    Set the model to training or evaluation mode.
    
    Args:
        model: PyTorch model
        training (bool): Whether to set to training mode
    """
    if training:
        model.train()
    else:
        model.eval()
