import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from cnnModel_pytorch import create_enhanced_model
from utils import (
    get_device_info, 
    create_dataloader, 
    prepare_batch_data, 
    calculate_output_vertices,
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    save_model,
    plot_training_curves
)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Get device information
device_info = get_device_info()
if device_info['cuda_available']:
    print(f"GPU: {device_info['gpu_name']}")
    print(f"GPU Memory: {device_info['gpu_memory']:.1f} GB")

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
        # This is a simplified version - in practice, you'd need more sophisticated symmetry detection
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

class EnhancedTrainModel:
    def __init__(self, img_path, mdl_path, obj_num, img_height=128, img_width=128, channels=3,
                 attention_heads=8, d_model=512):
        self.img_path = img_path
        self.mdl_path = mdl_path
        self.obj_num = obj_num
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.device = device
        
        # Create dataset and dataloader using utility functions
        self.dataloader, self.dataset = create_dataloader(img_path, mdl_path, obj_num, batch_size=16, shuffle=True)
        
        # Calculate output vertices from dataset
        self.output_vertices = calculate_output_vertices(self.dataset)
        print(f"Output vertices: {self.output_vertices}")
        
        # Create enhanced model with attention
        self.model = create_enhanced_model(
            img_height, img_width, channels, self.output_vertices,
            attention_heads=attention_heads, d_model=d_model, device=device
        )
        
        # Create optimizer and scheduler using utility functions
        self.optimizer = create_optimizer(self.model, lr=0.0001, optimizer_type='adam')  # Lower learning rate for complex model
        self.scheduler = create_scheduler(self.optimizer, mode='min', factor=0.7, patience=3, verbose=True)
        
        # Advanced loss function
        self.criterion = AdvancedLoss(vertex_weight=1.0, smoothness_weight=0.1, symmetry_weight=0.05)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.loss_components = []
        
    def train_epoch(self, epoch):
        """Train for one epoch with detailed loss tracking."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_loss_components = []
        
        for batch_idx, batch in enumerate(self.dataloader):
            try:
                images, targets = prepare_batch_data(batch, self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate advanced loss
                loss, loss_components = self.criterion(outputs, targets)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                epoch_loss_components.append(loss_components)
                
                if batch_idx % 5 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Total Loss: {loss.item():.6f}')
                    print(f'  Vertex Loss: {loss_components["vertex_loss"]:.6f}')
                    print(f'  Smoothness Loss: {loss_components["smoothness_loss"]:.6f}')
                    print(f'  Symmetry Loss: {loss_components["symmetry_loss"]:.6f}')
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
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
        """Validate the model with detailed loss tracking."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        val_loss_components = []
        
        with torch.no_grad():
            for batch in self.dataloader:
                try:
                    images, targets = prepare_batch_data(batch, self.device)
                    outputs = self.model(images)
                    loss, loss_components = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    val_loss_components.append(loss_components)
                    
                except Exception as e:
                    print(f"Validation error: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_loss_components = {
            'vertex_loss': np.mean([comp['vertex_loss'] for comp in val_loss_components]),
            'smoothness_loss': np.mean([comp['smoothness_loss'] for comp in val_loss_components]),
            'symmetry_loss': np.mean([comp['symmetry_loss'] for comp in val_loss_components]),
            'total_loss': avg_loss
        }
        
        return avg_loss, avg_loss_components
    
    def train_model(self, epochs=100):
        """Main training loop with enhanced monitoring."""
        print("Starting enhanced training with attention mechanisms...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
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
            
            # Print detailed progress
            print(f'\nEpoch {epoch + 1}/{epochs}:')
            print(f'  Training Loss: {train_loss:.6f}')
            print(f'  Validation Loss: {val_loss:.6f}')
            print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print('  Loss Components:')
            print(f'    Vertex: {train_components["vertex_loss"]:.6f} (train) / {val_components["vertex_loss"]:.6f} (val)')
            print(f'    Smoothness: {train_components["smoothness_loss"]:.6f} (train) / {val_components["smoothness_loss"]:.6f} (val)')
            print(f'    Symmetry: {train_components["symmetry_loss"]:.6f} (train) / {val_components["symmetry_loss"]:.6f} (val)')
            print('-' * 70)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                save_model(self.model, './final_models/enhanced_model_best.pth')
                print(f'  🎯 New best validation loss: {best_val_loss:.6f}')
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f'  🛑 Early stopping after {max_patience} epochs without improvement')
                    break
        
        # Save final model
        save_model(self.model, './final_models/enhanced_model_final.pth')
        
        # Plot enhanced training curves
        self.plot_enhanced_training_curves()
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint with enhanced information."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'loss_components': self.loss_components,
        }
        
        os.makedirs('./checkpoints', exist_ok=True)
        torch.save(checkpoint, f'./checkpoints/enhanced_model_epoch_{epoch:02d}.pth')
        print(f'Checkpoint saved: enhanced_model_epoch_{epoch:02d}.pth')
    
    def plot_enhanced_training_curves(self):
        """Plot enhanced training curves with loss components."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
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
        axes[1, 0].plot(train_smooth, label='Training Smoothness Loss', color='blue')
        axes[1, 0].plot(val_smooth, label='Validation Smoothness Loss', color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Smoothness Loss')
        axes[1, 0].set_title('Smoothness Loss Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Symmetry loss
        train_sym = [comp['train']['symmetry_loss'] for comp in self.loss_components]
        val_sym = [comp['val']['symmetry_loss'] for comp in self.loss_components]
        axes[1, 1].plot(train_sym, label='Training Symmetry Loss', color='blue')
        axes[1, 1].plot(val_sym, label='Validation Symmetry Loss', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Symmetry Loss')
        axes[1, 1].set_title('Symmetry Loss Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('./enhanced_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Configuration for enhanced model
    img_path = './shapes2d'
    mdl_path = './shapes3d'
    obj_num = 100  # Use more data for training
    
    img_height = 128
    img_width = 128
    channels = 3
    
    # Enhanced model parameters
    attention_heads = 8
    d_model = 512
    
    # Create enhanced trainer and start training
    trainer = EnhancedTrainModel(
        img_path, mdl_path, obj_num, img_height, img_width, channels,
        attention_heads=attention_heads, d_model=d_model
    )
    trainer.train_model(epochs=100)  # More epochs for complex model

if __name__ == "__main__":
    main()
