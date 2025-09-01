#!/usr/bin/env python3
"""
Installation Script for Hugging Face Integration

This script helps install the required dependencies and test the setup
for the Hugging Face dataset integration.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_pip():
    """Check if pip is available."""
    print("Checking pip availability...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✓ pip is available: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("✗ pip is not available. Please install pip first.")
        return False

def install_requirements():
    """Install requirements from requirements.txt."""
    # Look for requirements files
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    
    # Main requirements
    main_requirements = parent_dir / "requirements.txt"
    hf_requirements = current_dir / "requirements_hf.txt"
    
    if not main_requirements.exists():
        print("✗ Main requirements.txt not found. Please ensure you're in the correct directory.")
        return False
    
    if not hf_requirements.exists():
        print("✗ Hugging Face requirements file not found.")
        return False
    
    print("Installing main requirements...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        print("Warning: Failed to upgrade pip, continuing with installation...")
    
    # Install main requirements
    if not run_command(f"{sys.executable} -m pip install -r {main_requirements}", "Installing main requirements"):
        return False
    
    print("Installing Hugging Face requirements...")
    
    # Install Hugging Face requirements
    if not run_command(f"{sys.executable} -m pip install -r {hf_requirements}", "Installing Hugging Face requirements"):
        return False
    
    return True

def test_imports():
    """Test if the required libraries can be imported."""
    print("\nTesting library imports...")
    
    libraries = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("datasets", "Hugging Face Datasets"),
        ("huggingface_hub", "Hugging Face Hub"),
        ("transformers", "Transformers"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib"),
        ("cv2", "OpenCV"),
        ("trimesh", "Trimesh")
    ]
    
    all_imported = True
    
    for module_name, display_name in libraries:
        try:
            if module_name == "PIL":
                import PIL
                print(f"✓ {display_name} imported successfully")
            elif module_name == "cv2":
                import cv2
                print(f"✓ {display_name} imported successfully")
            else:
                module = __import__(module_name)
                version = getattr(module, "__version__", "unknown")
                print(f"✓ {display_name} imported successfully (version: {version})")
        except ImportError as e:
            print(f"✗ Failed to import {display_name}: {e}")
            all_imported = False
    
    return all_imported

def test_huggingface_integration():
    """Test the Hugging Face integration."""
    print("\nTesting Hugging Face integration...")
    
    try:
        # Test if our custom modules can be imported
        from huggingface_dataset_loader import DatasetManager, create_hf_dataloader
        print("✓ Custom dataset loader imported successfully")
        
        # Test dataset manager
        manager = DatasetManager()
        print("✓ Dataset manager created successfully")
        
        # List available datasets
        print("\nAvailable datasets:")
        manager.list_datasets()
        
        return True
        
    except ImportError as e:
        print(f"✗ Hugging Face integration test failed: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\nCreating necessary directories...")
    
    directories = [
        "./hf_cache",
        "./checkpoints", 
        "./final_models",
        "./logs",
        "./plots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def run_integration_test():
    """Run the integration test script."""
    print("\nRunning integration test...")
    
    test_script = Path("test_hf_integration.py")
    if not test_script.exists():
        print("✗ test_hf_integration.py not found")
        return False
    
    try:
        result = subprocess.run([sys.executable, "test_hf_integration.py"], 
                              capture_output=True, text=True, check=True)
        print("✓ Integration test completed successfully")
        print("\nTest output:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Integration test failed with error code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    """Main installation function."""
    print("=" * 60)
    print("Hugging Face Integration Installation Script")
    print("=" * 60)
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    if not check_pip():
        return False
    
    # Install requirements
    if not install_requirements():
        print("\n✗ Installation failed. Please check the errors above.")
        return False
    
    # Test imports
    if not test_imports():
        print("\n⚠️  Some libraries failed to import. This may cause issues.")
    
    # Test Hugging Face integration
    if not test_huggingface_integration():
        print("\n✗ Hugging Face integration test failed.")
        return False
    
    # Create directories
    if not create_directories():
        print("\n✗ Failed to create directories.")
        return False
    
    # Run integration test
    if not run_integration_test():
        print("\n⚠️  Integration test failed. Please check the errors above.")
    
    # Success message
    print("\n" + "=" * 60)
    print("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Start training: python train_with_huggingface.py")
    print("2. View help: python train_with_huggingface.py --help")
    print("3. Test with small dataset: python train_with_huggingface.py --max-samples 100 --epochs 5")
    
    print("\nAvailable datasets:")
    print("- modelnet: Quick testing (recommended for first run)")
    print("- shapenet: Large-scale training")
    print("- partnet: Fine-grained parts")
    print("- scannet: Indoor scenes")
    
    print("\nFor more information, see README_HuggingFace.md")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInstallation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during installation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
