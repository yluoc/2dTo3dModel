"""
Script to capture 2D images from all 3D models in the training dataset.

This script processes all 3D models in the training_dataset/3dModel directory
and generates 2D images from multiple angles, saving them to training_dataset/2dImg.
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from utils.frame_capture import ModelFrameCapture

def main():
    """
    Main function to capture frames from all 3D models.
    """
    print("3D Model Frame Capture - Processing All Models")
    print("=" * 50)
    
    # Initialize the frame capture utility
    capture_util = ModelFrameCapture(
        output_dir="./training_dataset/2dImg",
        image_size=(512, 512),           # High resolution images
        num_angles=12,                   # 12 azimuth angles
        elevation_range=(10, 80),        # 10 to 80 degrees elevation
        distance_factor=2.0              # Camera distance multiplier
    )
    
    print(f"Configuration:")
    print(f"  - Output directory: {capture_util.output_dir}")
    print(f"  - Image size: {capture_util.image_size}")
    print(f"  - Number of azimuth angles: {capture_util.num_angles}")
    print(f"  - Elevation range: {capture_util.elevation_range}")
    print(f"  - Distance factor: {capture_util.distance_factor}")
    print(f"  - Total frames per model: {capture_util.num_angles * 3}")  # 3 elevation levels
    print()
    
    # Process all models
    capture_util.process_all_models("./training_dataset/3dModel")
    
    print("\n" + "=" * 50)
    print("Frame capture process completed!")
    print(f"Check the output directory: {capture_util.output_dir}")

if __name__ == "__main__":
    main()
