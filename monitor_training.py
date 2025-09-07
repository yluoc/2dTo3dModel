"""
Training monitoring script.

This script helps monitor the training progress and provides status updates.
"""

import os
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt

def check_training_status():
    """Check the current training status."""
    print("🔍 Training Status Check")
    print("=" * 50)
    
    # Check if training is running
    training_log = Path("training.log")
    if training_log.exists():
        print(f"✅ Training log found: {training_log}")
        
        # Read last few lines
        with open(training_log, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"📝 Last log entry: {lines[-1].strip()}")
            else:
                print("📝 Training log is empty")
    else:
        print("❌ No training log found")
    
    # Check for checkpoints
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        checkpoint_files = list(checkpoints_dir.glob("*.pth"))
        if checkpoint_files:
            print(f"💾 Found {len(checkpoint_files)} checkpoint(s)")
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            print(f"📁 Latest checkpoint: {latest_checkpoint.name}")
        else:
            print("💾 No checkpoints found")
    else:
        print("❌ No checkpoints directory")
    
    # Check for final models
    models_dir = Path("final_models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth"))
        if model_files:
            print(f"🎯 Found {len(model_files)} final model(s)")
            for model_file in model_files:
                print(f"   - {model_file.name}")
        else:
            print("🎯 No final models found")
    else:
        print("❌ No final models directory")
    
    # Check for training history
    history_file = Path("logs/training_history.json")
    if history_file.exists():
        print(f"📊 Training history found: {history_file}")
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
                if 'train_losses' in history:
                    print(f"📈 Training epochs completed: {len(history['train_losses'])}")
                    if history['train_losses']:
                        print(f"📉 Latest training loss: {history['train_losses'][-1]:.6f}")
                if 'val_losses' in history:
                    if history['val_losses']:
                        print(f"📉 Latest validation loss: {history['val_losses'][-1]:.6f}")
        except Exception as e:
            print(f"❌ Error reading training history: {e}")
    else:
        print("📊 No training history found")

def plot_training_curves():
    """Plot training curves if available."""
    history_file = Path("logs/training_history.json")
    if not history_file.exists():
        print("❌ No training history found for plotting")
        return
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        if 'train_losses' not in history or 'val_losses' not in history:
            print("❌ Incomplete training history")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot total losses
        plt.subplot(2, 2, 1)
        plt.plot(history['train_losses'], label='Training Loss', color='blue')
        plt.plot(history['val_losses'], label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot loss components if available
        if 'loss_components' in history and history['loss_components']:
            components = history['loss_components']
            
            # Vertex loss
            plt.subplot(2, 2, 2)
            train_vertex = [comp['train']['vertex_loss'] for comp in components]
            val_vertex = [comp['val']['vertex_loss'] for comp in components]
            plt.plot(train_vertex, label='Training', color='blue')
            plt.plot(val_vertex, label='Validation', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Vertex Loss')
            plt.title('Vertex Loss')
            plt.legend()
            plt.grid(True)
            
            # Smoothness loss
            plt.subplot(2, 2, 3)
            train_smooth = [comp['train']['smoothness_loss'] for comp in components]
            val_smooth = [comp['val']['smoothness_loss'] for comp in components]
            plt.plot(train_smooth, label='Training', color='blue')
            plt.plot(val_smooth, label='Validation', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Smoothness Loss')
            plt.title('Smoothness Loss')
            plt.legend()
            plt.grid(True)
            
            # Chamfer loss
            plt.subplot(2, 2, 4)
            train_chamfer = [comp['train']['chamfer_loss'] for comp in components]
            val_chamfer = [comp['val']['chamfer_loss'] for comp in components]
            plt.plot(train_chamfer, label='Training', color='blue')
            plt.plot(val_chamfer, label='Validation', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Chamfer Loss')
            plt.title('Chamfer Loss')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_status.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Training curves saved as 'training_status.png'")
        
    except Exception as e:
        print(f"❌ Error plotting training curves: {e}")

def main():
    """Main monitoring function."""
    print("🚀 Training Monitor")
    print("=" * 50)
    
    while True:
        check_training_status()
        print("\n" + "=" * 50)
        
        choice = input("Options:\n1. Plot training curves\n2. Refresh status\n3. Exit\nChoice (1-3): ")
        
        if choice == '1':
            plot_training_curves()
        elif choice == '2':
            continue
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
