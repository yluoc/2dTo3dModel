"""
Training script for 2D to 3D model reconstruction using generated dataset.

This script trains a deep learning model to reconstruct 3D models from 2D images
using the generated training dataset with multiple viewpoints.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.cnnModel_pytorch import create_enhanced_model
from src.utils import (
    create_2d3d_dataloader,
    get_dataset_info_2d3d,
    split_dataset,
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    save_model,
    get_device_info
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Get device information
device_info = get_device_info()
if device_info['cuda_available']:
    logger.info(f"GPU: {device_info['gpu_name']}")
    logger.info(f"GPU Memory: {device_info['gpu_memory']:.1f} GB")


class AdvancedLoss3D(nn.Module):
    """
    Advanced loss function for 3D reconstruction with multiple components.
    """
    
    def __init__(self, 
                 vertex_weight=1.0, 
                 smoothness_weight=0.1, 
                 symmetry_weight=0.05,
                 chamfer_weight=0.1):
        super().__init__()
        self.vertex_weight = vertex_weight
        self.smoothness_weight = smoothness_weight
        self.symmetry_weight = symmetry_weight
        self.chamfer_weight = chamfer_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def vertex_loss(self, pred_vertices, target_vertices):
        """Standard vertex coordinate loss."""
        return self.mse_loss(pred_vertices, target_vertices)
    
    def smoothness_loss(self, pred_vertices):
        """Smoothness loss to encourage smooth 3D surfaces."""
        batch_size = pred_vertices.size(0)
        vertices = pred_vertices.view(batch_size, -1, 3)
        
        # Calculate differences between adjacent vertices
        vertex_diff = vertices[:, 1:, :] - vertices[:, :-1, :]
        smoothness = torch.mean(torch.norm(vertex_diff, dim=2))
        
        return smoothness
    
    def symmetry_loss(self, pred_vertices):
        """Symmetry loss to encourage symmetric 3D shapes."""
        batch_size = pred_vertices.size(0)
        vertices = pred_vertices.view(batch_size, -1, 3)
        
        # For simplicity, assume vertices are ordered and we can find symmetric pairs
        mid_point = vertices.size(1) // 2
        
        if vertices.size(1) % 2 == 0:
            # Even number of vertices
            left_vertices = vertices[:, :mid_point, :]
            right_vertices = torch.flip(vertices[:, mid_point:, :], dims=[1])
            
            # Flip x-coordinate for symmetry
            right_vertices[:, :, 0] = -right_vertices[:, :, 0]
            
            symmetry_loss = self.mse_loss(left_vertices, right_vertices)
        else:
            # Odd number of vertices
            symmetry_loss = torch.tensor(0.0, device=vertices.device)
        
        return symmetry_loss
    
    def chamfer_loss(self, pred_vertices, target_vertices):
        """Chamfer distance loss for better 3D reconstruction."""
        batch_size = pred_vertices.size(0)
        pred_vertices = pred_vertices.view(batch_size, -1, 3)
        target_vertices = target_vertices.view(batch_size, -1, 3)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(pred_vertices, target_vertices)
        
        # Chamfer distance: sum of minimum distances
        chamfer_dist = torch.mean(torch.min(dist_matrix, dim=2)[0]) + \
                      torch.mean(torch.min(dist_matrix, dim=1)[0])
        
        return chamfer_dist
    
    def forward(self, pred_vertices, target_vertices):
        """Combined loss function."""
        vertex_loss = self.vertex_loss(pred_vertices, target_vertices)
        smoothness_loss = self.smoothness_loss(pred_vertices)
        symmetry_loss = self.symmetry_loss(pred_vertices)
        chamfer_loss = self.chamfer_loss(pred_vertices, target_vertices)
        
        total_loss = (self.vertex_weight * vertex_loss + 
                     self.smoothness_weight * smoothness_loss + 
                     self.symmetry_weight * symmetry_loss +
                     self.chamfer_weight * chamfer_loss)
        
        return total_loss, {
            'vertex_loss': vertex_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'symmetry_loss': symmetry_loss.item(),
            'chamfer_loss': chamfer_loss.item(),
            'total_loss': total_loss.item()
        }


class Trainer2D3D:
    """
    Trainer class for 2D to 3D model reconstruction.
    """
    
    def __init__(self, 
                 img_dir: str = "./training_dataset/2dImg",
                 model_dir: str = "./training_dataset/3dModel",
                 batch_size: int = 8,
                 learning_rate: float = 0.0001,
                 max_vertices: int = 10000,
                 attention_heads: int = 8,
                 d_model: int = 512):
        """
        Initialize the trainer.
        
        Args:
            img_dir: Directory containing 2D images
            model_dir: Directory containing 3D models
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            max_vertices: Maximum number of vertices per model
            attention_heads: Number of attention heads
            d_model: Model dimension
        """
        self.img_dir = img_dir
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_vertices = max_vertices
        self.device = device
        
        # Get dataset information
        self.dataset_info = get_dataset_info_2d3d(img_dir, model_dir)
        logger.info(f"Dataset info: {self.dataset_info}")
        
        # Create dataloaders
        self.train_loader, self.train_dataset = create_2d3d_dataloader(
            img_dir=img_dir,
            model_dir=model_dir,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            max_vertices=max_vertices,
            augment=True
        )
        
        # Split dataset for validation
        train_dataset, val_dataset = split_dataset(self.train_dataset, train_ratio=0.8)
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.train_loader.collate_fn
        )
        
        # Create model with fixed output dimension
        fixed_output_dim = max_vertices * 3  # 2000 vertices * 3 coordinates = 6000
        self.model = create_enhanced_model(
            img_height=512,
            img_width=512,
            channels=3,
            output_vertices=fixed_output_dim,  # Fixed output dimension
            attention_heads=attention_heads,
            d_model=d_model,
            device=device
        )
        
        # Create optimizer and scheduler
        self.optimizer = create_optimizer(self.model, lr=learning_rate, optimizer_type='adam')
        self.scheduler = create_scheduler(self.optimizer, mode='min', factor=0.7, patience=5, verbose=True)
        
        # Loss function
        self.criterion = AdvancedLoss3D(
            vertex_weight=1.0,
            smoothness_weight=0.1,
            symmetry_weight=0.05,
            chamfer_weight=0.1
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.loss_components = []
        
        # Create output directories
        os.makedirs('./checkpoints', exist_ok=True)
        os.makedirs('./final_models', exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_loss_components = []
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            try:
                # Move data to device
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss, loss_components = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                epoch_loss_components.append(loss_components)
                
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_loss_components = {
            'vertex_loss': np.mean([comp['vertex_loss'] for comp in epoch_loss_components]),
            'smoothness_loss': np.mean([comp['smoothness_loss'] for comp in epoch_loss_components]),
            'symmetry_loss': np.mean([comp['symmetry_loss'] for comp in epoch_loss_components]),
            'chamfer_loss': np.mean([comp['chamfer_loss'] for comp in epoch_loss_components]),
            'total_loss': avg_loss
        }
        
        return avg_loss, avg_loss_components
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        val_loss_components = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                try:
                    # Move data to device
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    
                    # Calculate loss
                    loss, loss_components = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    val_loss_components.append(loss_components)
                    
                except Exception as e:
                    logger.error(f"Validation error: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_loss_components = {
            'vertex_loss': np.mean([comp['vertex_loss'] for comp in val_loss_components]),
            'smoothness_loss': np.mean([comp['smoothness_loss'] for comp in val_loss_components]),
            'symmetry_loss': np.mean([comp['symmetry_loss'] for comp in val_loss_components]),
            'chamfer_loss': np.mean([comp['chamfer_loss'] for comp in val_loss_components]),
            'total_loss': avg_loss
        }
        
        return avg_loss, avg_loss_components
    
    def train(self, epochs: int = 100):
        """Main training loop."""
        logger.info("Starting 2D to 3D model training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(epochs):
            # Training
            train_loss, train_components = self.train_epoch(epoch + 1)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_components = self.validate()
            self.val_losses.append(val_loss)
            
            # Store loss components
            self.loss_components.append({
                'train': train_components,
                'val': val_components
            })
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Log progress
            logger.info(f'Epoch {epoch + 1}/{epochs}:')
            logger.info(f'  Train Loss: {train_loss:.6f}')
            logger.info(f'  Val Loss: {val_loss:.6f}')
            logger.info(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_model(self.model, './final_models/2d3d_model_best.pth')
                logger.info(f'  🎯 New best validation loss: {best_val_loss:.6f}')
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f'  🛑 Early stopping after {max_patience} epochs without improvement')
                    break
        
        # Save final model
        save_model(self.model, './final_models/2d3d_model_final.pth')
        
        # Plot training curves
        self.plot_training_curves()
        
        # Save training history
        self.save_training_history()
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'loss_components': self.loss_components,
            'dataset_info': self.dataset_info
        }
        
        torch.save(checkpoint, f'./checkpoints/2d3d_model_epoch_{epoch:03d}.pth')
        logger.info(f'Checkpoint saved: 2d3d_model_epoch_{epoch:03d}.pth')
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Total loss
        axes[0, 0].plot(self.train_losses, label='Training Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Vertex loss
        train_vertex = [comp['train']['vertex_loss'] for comp in self.loss_components]
        val_vertex = [comp['val']['vertex_loss'] for comp in self.loss_components]
        axes[0, 1].plot(train_vertex, label='Training Vertex Loss', color='blue')
        axes[0, 1].plot(val_vertex, label='Validation Vertex Loss', color='red')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Vertex Loss')
        axes[0, 1].set_title('Vertex Loss Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Smoothness loss
        train_smooth = [comp['train']['smoothness_loss'] for comp in self.loss_components]
        val_smooth = [comp['val']['smoothness_loss'] for comp in self.loss_components]
        axes[0, 2].plot(train_smooth, label='Training Smoothness Loss', color='blue')
        axes[0, 2].plot(val_smooth, label='Validation Smoothness Loss', color='red')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Smoothness Loss')
        axes[0, 2].set_title('Smoothness Loss Over Time')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Symmetry loss
        train_sym = [comp['train']['symmetry_loss'] for comp in self.loss_components]
        val_sym = [comp['val']['symmetry_loss'] for comp in self.loss_components]
        axes[1, 0].plot(train_sym, label='Training Symmetry Loss', color='blue')
        axes[1, 0].plot(val_sym, label='Validation Symmetry Loss', color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Symmetry Loss')
        axes[1, 0].set_title('Symmetry Loss Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Chamfer loss
        train_chamfer = [comp['train']['chamfer_loss'] for comp in self.loss_components]
        val_chamfer = [comp['val']['chamfer_loss'] for comp in self.loss_components]
        axes[1, 1].plot(train_chamfer, label='Training Chamfer Loss', color='blue')
        axes[1, 1].plot(val_chamfer, label='Validation Chamfer Loss', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Chamfer Loss')
        axes[1, 1].set_title('Chamfer Loss Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Learning rate
        lr_history = [group['lr'] for group in self.optimizer.param_groups]
        axes[1, 2].plot(lr_history, label='Learning Rate', color='green')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('./logs/2d3d_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_training_history(self):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'loss_components': self.loss_components,
            'dataset_info': self.dataset_info,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('./logs/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("Training history saved to logs/training_history.json")


def main():
    """Main function to run training."""
    # Configuration optimized for sphere dataset
    config = {
        'img_dir': './training_dataset/2dImg',
        'model_dir': './training_dataset/3dModel',
        'batch_size': 8,  # Increased for sphere models (simpler geometry)
        'learning_rate': 0.0001,
        'max_vertices': 2000,  # Reduced for sphere models (simpler geometry)
        'attention_heads': 4,  # Reduced for simpler models
        'd_model': 256,  # Reduced for simpler models
        'epochs': 50  # Reduced for faster training
    }
    
    epochs = config.pop('epochs')  # Remove epochs from config
    
    logger.info(f"Training configuration: {config}")
    logger.info(f"Training epochs: {epochs}")
    
    # Create trainer
    trainer = Trainer2D3D(**config)
    
    # Start training
    trainer.train(epochs=epochs)


if __name__ == "__main__":
    main()
