"""
Test Script for Hugging Face Dataset Integration

This script tests the Hugging Face dataset loader to ensure it works correctly
before running the full training pipeline.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_huggingface_import():
    """Test if Hugging Face libraries can be imported."""
    print("Testing Hugging Face library imports...")
    
    try:
        import datasets
        print(f"✓ datasets library imported successfully (version: {datasets.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import datasets library: {e}")
        return False
    
    try:
        import huggingface_hub
        print(f"✓ huggingface_hub library imported successfully (version: {huggingface_hub.__version__})")
        return True
    except ImportError as e:
        print(f"✗ Failed to import huggingface_hub library: {e}")
        return False

def test_dataset_loader_import():
    """Test if the custom dataset loader can be imported."""
    print("\nTesting custom dataset loader import...")
    
    try:
        from huggingface_dataset_loader import (
            HuggingFace2D3DDataset,
            DatasetManager,
            create_hf_dataloader
        )
        print("✓ Custom dataset loader imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import custom dataset loader: {e}")
        return False

def test_dataset_manager():
    """Test the dataset manager functionality."""
    print("\nTesting dataset manager...")
    
    try:
        from huggingface_dataset_loader import DatasetManager
        
        manager = DatasetManager()
        print("✓ Dataset manager created successfully")
        
        # List available datasets
        print("\nAvailable datasets:")
        manager.list_datasets()
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset manager test failed: {e}")
        return False

def test_small_dataset_loading():
    """Test loading a small dataset for verification."""
    print("\nTesting small dataset loading...")
    
    try:
        from huggingface_dataset_loader import create_hf_dataloader
        
        # Try to load a very small dataset
        print("Attempting to load 5 samples from ModelNet dataset...")
        
        dataloader, dataset = create_hf_dataloader(
            dataset_key="modelnet",
            split="train",
            max_samples=5,
            batch_size=2
        )
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Dataset size: {len(dataset)}")
        print(f"  - Dataloader batches: {len(dataloader)}")
        
        # Test a single batch
        print("\nTesting batch iteration...")
        for i, batch in enumerate(dataloader):
            print(f"  Batch {i + 1}:")
            print(f"    - Images shape: {batch['images'].shape}")
            print(f"    - Vertices shape: {batch['vertices'].shape}")
            print(f"    - Faces shape: {batch['faces'].shape}")
            if i >= 1:  # Just test first 2 batches
                break
        
        return True
        
    except Exception as e:
        print(f"✗ Small dataset loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script_import():
    """Test if the training script can be imported."""
    print("\nTesting training script import...")
    
    try:
        from train_with_huggingface import HuggingFaceTrainer, create_default_config
        print("✓ Training script imported successfully")
        
        # Test configuration creation
        config = create_default_config()
        print(f"✓ Default configuration created with {len(config)} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Training script import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Hugging Face Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Hugging Face Libraries", test_huggingface_import),
        ("Dataset Loader Import", test_dataset_loader_import),
        ("Dataset Manager", test_dataset_manager),
        ("Small Dataset Loading", test_small_dataset_loading),
        ("Training Script Import", test_training_script_import)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Hugging Face integration is ready to use.")
        print("\nNext steps:")
        print("1. Run: python train_with_huggingface.py")
        print("2. Or customize training with: python train_with_huggingface.py --help")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check internet connection for dataset downloads")
        print("3. Verify Python environment and imports")

if __name__ == "__main__":
    main()
