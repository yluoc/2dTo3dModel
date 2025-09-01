"""
Enhanced Training Script with Hugging Face Dataset Integration

This script extends the existing training pipeline to use Hugging Face datasets
for 2D to 3D model training. It provides a unified interface for both local
and Hugging Face datasets.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from pathlib import Path

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import existing modules
from cnnModel_pytorch import create_enhanced_model
from utils import (
    get_device_info, 
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    save_model,
    plot_training_curves
)

# Import Hugging Face dataset loader from current directory
try:
    from huggingface_dataset_loader import (
        create_hf_dataloader, 
        DatasetManager,
        HuggingFace2D3DDataset
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: Hugging Face dataset loader not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedLoss(nn.Module):
    """
    Advanced loss function combining multiple loss components for better 3D reconstruction.
    """
    def __init__(self, vertex_weight=1.0, smoothness_weight=0.1, symmetry_weight=0.05):
        super().__init__()
        self.vertex_weight = vertex_weight
        self.smoothness_weight = smoothness_weight
        self.symmetry_weight = symmetry_weight
        
        self.mse_loss = nn.MSELoss()
        
    def vertex_loss(self, pred_vertices, target_vertices):
        """Standard vertex coordinate loss."""
        return self.mse_loss(pred_vertices, target_vertices)
    
    def smoothness_loss(self, pred_vertices):
        """Smoothness loss to encourage smooth 3D surfaces."""
        # Reshape to (batch, vertices, 3)
        batch_size = pred_vertices.size(0)
        vertices = pred_vertices.view(batch_size, -1, 3)
        
        # Calculate differences between adjacent vertices
        vertex_diff = vertices[:, 1:, :] - vertices[:, :-1, :]
        smoothness = torch.mean(torch.norm(vertex_diff, dim=2))
        
        return smoothness
    
    def symmetry_loss(self, pred_vertices):
        """Symmetry loss to encourage symmetric 3D shapes."""
        # Reshape to (batch, vertices, 3)
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
    
    def forward(self, pred_vertices, target_vertices):
        """Combined loss function."""
        vertex_loss = self.vertex_loss(pred_vertices, target_vertices)
        smoothness_loss = self.smoothness_loss(pred_vertices)
        symmetry_loss = self.symmetry_loss(pred_vertices)
        
        total_loss = (self.vertex_weight * vertex_loss + 
                     self.smoothness_weight * smoothness_loss + 
                     self.symmetry_weight * symmetry_loss)
        
        return total_loss, {
            'vertex_loss': vertex_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'symmetry_loss': symmetry_loss.item(),
            'total_loss': total_loss.item()
        }


class HuggingFaceTrainer:
    """
    Enhanced trainer class that supports both local and Hugging Face datasets.
    """
    
    def __init__(self, config):
        """
        Initialize the trainer.
        
        Args:
            config (dict): Training configuration
        """
        self.config = config
        self.device = self._setup_device()
        
        # Initialize model, optimizer, and loss
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.loss_components = []
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def _setup_device(self):
        """Setup and return the training device."""
        device_info = get_device_info()
        device = torch.device(device_info['device'])
        
        if device_info['cuda_available']:
            logger.info(f"GPU: {device_info['gpu_name']}")
            logger.info(f"GPU Memory: {device_info['gpu_memory']:.1f} GB")
        else:
            logger.info("Using CPU for training")
        
        return device
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.get('log_dir', './logs'))
        log_dir.mkdir(exist_ok=True)
        
        # Create experiment log file
        experiment_name = self.config.get('experiment_name', 'hf_training')
        self.log_file = log_dir / f"{experiment_name}.log"
        
        # Add file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    def setup_data(self):
        """Setup training and validation data."""
        logger.info("Setting up training data...")
        
        if self.config.get('use_huggingface', False):
            if not HF_AVAILABLE:
                raise ImportError("Hugging Face integration not available")
            
            # Load Hugging Face datasets
            self.train_dataloader, self.train_dataset = create_hf_dataloader(
                dataset_key=self.config['hf_dataset'],
                split='train',
                batch_size=self.config['batch_size'],
                max_samples=self.config.get('max_samples'),
                cache_dir=self.config.get('hf_cache_dir', './hf_cache')
            )
            
            self.val_dataloader, self.val_dataset = create_hf_dataloader(
                dataset_key=self.config['hf_dataset'],
                split='validation',
                batch_size=self.config['batch_size'],
                max_samples=self.config.get('max_val_samples'),
                cache_dir=self.config.get('hf_cache_dir', './hf_cache')
            )
            
            # Get output dimensions from dataset
            sample_batch = next(iter(self.train_dataloader))
            self.output_vertices = sample_batch['vertices'].size(1)
            
            logger.info(f"Loaded Hugging Face dataset: {self.config['hf_dataset']}")
            logger.info(f"Training samples: {len(self.train_dataset)}")
            logger.info(f"Validation samples: {len(self.val_dataset)}")
            logger.info(f"Output vertices: {self.output_vertices}")
            
        else:
            # Use local dataset (existing functionality)
            from utils import create_dataloader, calculate_output_vertices
            
            self.train_dataloader, self.train_dataset = create_dataloader(
                self.config['img_path'],
                self.config['mdl_path'],
                self.config['obj_num'],
                batch_size=self.config['batch_size'],
                shuffle=True
            )
            
            self.val_dataloader, self.val_dataset = create_dataloader(
                self.config['img_path'],
                self.config['mdl_path'],
                self.config['obj_num'],
                batch_size=self.config['batch_size'],
                shuffle=False
            )
            
            self.output_vertices = calculate_output_vertices(self.train_dataset)
            logger.info(f"Loaded local dataset with {self.output_vertices} output vertices")
    
    def setup_model(self):
        """Setup the neural network model."""
        logger.info("Setting up neural network model...")
        
        self.model = create_enhanced_model(
            self.config['img_height'],
            self.config['img_width'],
            self.config['channels'],
            self.output_vertices,
            attention_heads=self.config.get('attention_heads', 8),
            d_model=self.config.get('d_model', 512),
            device=self.device
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, and loss function."""
        logger.info("Setting up training components...")
        
        # Optimizer
        self.optimizer = create_optimizer(
            self.model,
            lr=self.config.get('learning_rate', 0.0001),
            optimizer_type=self.config.get('optimizer', 'adam')
        )
        
        # Scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            mode='min',
            factor=self.config.get('scheduler_factor', 0.7),
            patience=self.config.get('scheduler_patience', 3),
            verbose=True
        )
        
        # Loss function
        self.criterion = AdvancedLoss(
            vertex_weight=self.config.get('vertex_weight', 1.0),
            smoothness_weight=self.config.get('smoothness_weight', 0.1),
            symmetry_weight=self.config.get('symmetry_weight', 0.05)
        )
        
        logger.info("Training components setup complete")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_loss_components = []
        
        progress_bar = tqdm(self.train_dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Prepare batch data
                if self.config.get('use_huggingface', False):
                    images = batch['images'].to(self.device)
                    targets = batch['vertices'].to(self.device)
                else:
                    from utils import prepare_batch_data
                    images, targets = prepare_batch_data(batch, self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss, loss_components = self.criterion(outputs, targets)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                epoch_loss_components.append(loss_components)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Vertex': f'{loss_components["vertex_loss"]:.6f}',
                    'Smooth': f'{loss_components["smoothness_loss"]:.6f}'
                })
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_loss_components = {
            'vertex_loss': np.mean([comp['vertex_loss'] for comp in epoch_loss_components]),
            'smoothness_loss': np.mean([comp['smoothness_loss'] for comp in epoch_loss_components]),
            'symmetry_loss': np.mean([comp['symmetry_loss'] for comp in epoch_loss_components]),
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
            for batch in tqdm(self.val_dataloader, desc='Validation'):
                try:
                    # Prepare batch data
                    if self.config.get('use_huggingface', False):
                        images = batch['images'].to(self.device)
                        targets = batch['vertices'].to(self.device)
                    else:
                        from utils import prepare_batch_data
                        images, targets = prepare_batch_data(batch, self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    
                    # Calculate loss
                    loss, loss_components = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    val_loss_components.append(loss_components)
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_loss_components = {
            'vertex_loss': np.mean([comp['vertex_loss'] for comp in val_loss_components]),
            'smoothness_loss': np.mean([comp['smoothness_loss'] for comp in val_loss_components]),
            'smoothness_loss': np.mean([comp['smoothness_loss'] for comp in val_loss_components]),
            'total_loss': avg_loss
        }
        
        return avg_loss, avg_loss_components
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with loss: {loss:.6f}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup components
        self.setup_data()
        self.setup_model()
        self.setup_training_components()
        
        # Training loop
        num_epochs = self.config.get('num_epochs', 100)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss, train_components = self.train_epoch(epoch + 1)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_components = self.validate()
            self.val_losses.append(val_loss)
            
            # Log results
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            logger.info(f"  Train Components: {train_components}")
            logger.info(f"  Val Components: {val_components}")
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            self.save_checkpoint(epoch + 1, val_loss, is_best)
            
            # Early stopping check
            if self.config.get('early_stopping', False):
                patience = self.config.get('patience', 10)
                if len(self.val_losses) > patience:
                    recent_losses = self.val_losses[-patience:]
                    if all(loss >= recent_losses[0] for loss in recent_losses):
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
        
        # Save final model
        final_model_path = Path(self.config.get('final_model_dir', './final_models')) / 'hf_trained_model.pth'
        final_model_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'final_loss': self.best_loss
        }, final_model_path)
        
        logger.info(f"Training complete. Final model saved to {final_model_path}")
        
        # Plot training curves
        self.plot_training_results()
    
    def plot_training_results(self):
        """Plot training and validation curves."""
        try:
            plt.figure(figsize=(15, 5))
            
            # Loss curves
            plt.subplot(1, 3, 1)
            plt.plot(self.train_losses, label='Training Loss', color='blue')
            plt.plot(self.val_losses, label='Validation Loss', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            # Learning rate
            plt.subplot(1, 3, 2)
            lr_history = [group['lr'] for group in self.optimizer.param_groups]
            plt.plot(lr_history, label='Learning Rate', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True)
            
            # Loss components (last epoch)
            if self.loss_components:
                plt.subplot(1, 3, 3)
                components = self.loss_components[-1]
                component_names = list(components.keys())
                component_values = list(components.values())
                
                plt.bar(component_names, component_values, color=['blue', 'orange', 'green', 'red'])
                plt.xlabel('Loss Components')
                plt.ylabel('Loss Value')
                plt.title('Loss Components (Last Epoch)')
                plt.xticks(rotation=45)
                plt.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = Path(self.config.get('plot_dir', './plots')) / 'training_results.png'
            plot_path.parent.mkdir(exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Training plots saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting training results: {e}")


def create_default_config():
    """Create a default training configuration."""
    return {
        # Dataset configuration
        'use_huggingface': True,
        'hf_dataset': 'modelnet',  # Use ModelNet as default (smaller, faster)
        'max_samples': 1000,  # Limit samples for faster training
        'max_val_samples': 200,
        'hf_cache_dir': './hf_cache',
        
        # Model configuration
        'img_height': 128,
        'img_width': 128,
        'channels': 3,
        'attention_heads': 8,
        'd_model': 512,
        
        # Training configuration
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 0.0001,
        'optimizer': 'adam',
        'scheduler_factor': 0.7,
        'scheduler_patience': 3,
        
        # Loss weights
        'vertex_weight': 1.0,
        'smoothness_weight': 0.1,
        'symmetry_weight': 0.05,
        
        # Training options
        'early_stopping': True,
        'patience': 10,
        
        # Paths
        'checkpoint_dir': './checkpoints',
        'final_model_dir': './final_models',
        'log_dir': './logs',
        'plot_dir': './plots',
        'experiment_name': 'hf_training'
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train 2D to 3D model with Hugging Face datasets')
    
    # Dataset options
    parser.add_argument('--use-hf', action='store_true', default=True,
                       help='Use Hugging Face datasets (default: True)')
    parser.add_argument('--hf-dataset', type=str, default='modelnet',
                       choices=['shapenet', 'modelnet', 'partnet', 'scannet'],
                       help='Hugging Face dataset to use')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum number of training samples')
    parser.add_argument('--max-val-samples', type=int, default=200,
                       help='Maximum number of validation samples')
    
    # Model options
    parser.add_argument('--attention-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--d-model', type=int, default=512,
                       help='Model dimension')
    
    # Training options
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create configuration
    config = create_default_config()
    
    # Update with command line arguments
    config.update({
        'use_huggingface': args.use_hf,
        'hf_dataset': args.hf_dataset,
        'max_samples': args.max_samples,
        'max_val_samples': args.max_val_samples,
        'attention_heads': args.attention_heads,
        'd_model': args.d_model,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr
    })
    
    # Create and run trainer
    trainer = HuggingFaceTrainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
