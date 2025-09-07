"""
3D Model Frame Capture Utility

This module provides functionality to capture 2D images of 3D models from different angles
for training dataset generation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from PIL import Image
import math
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFrameCapture:
    """
    A class to capture 2D images of 3D models from different camera angles.
    """
    
    def __init__(self, 
                 output_dir: str = "./training_dataset/2dImg",
                 image_size: Tuple[int, int] = (512, 512),
                 num_angles: int = 12,
                 elevation_range: Tuple[float, float] = (10, 80),
                 distance_factor: float = 2.0):
        """
        Initialize the frame capture utility.
        
        Args:
            output_dir: Directory to save captured images
            image_size: Size of output images (width, height)
            num_angles: Number of azimuth angles to capture
            elevation_range: Range of elevation angles (min, max) in degrees
            distance_factor: Factor to determine camera distance from model center
        """
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.num_angles = num_angles
        self.elevation_range = elevation_range
        self.distance_factor = distance_factor
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_3d_model(self, model_path: str) -> Optional[trimesh.Trimesh]:
        """
        Load a 3D model from file.
        
        Args:
            model_path: Path to the 3D model file (.obj, .ply, .stl, etc.)
            
        Returns:
            Trimesh object or None if loading fails
        """
        try:
            loaded = trimesh.load(model_path)
            
            # Handle different return types from trimesh.load()
            if isinstance(loaded, trimesh.Scene):
                # If it's a scene, try to get the first mesh
                if len(loaded.geometry) > 0:
                    mesh = list(loaded.geometry.values())[0]
                    if isinstance(mesh, trimesh.Trimesh):
                        logger.info(f"Successfully loaded model from scene: {model_path}")
                        return mesh
                    else:
                        logger.warning(f"Scene contains non-mesh geometry: {model_path}")
                        return None
                else:
                    logger.warning(f"Scene is empty: {model_path}")
                    return None
            elif isinstance(loaded, trimesh.Trimesh):
                logger.info(f"Successfully loaded model: {model_path}")
                return loaded
            else:
                logger.warning(f"Unknown geometry type loaded: {type(loaded)}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            return None
    
    def normalize_model(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Normalize the model to fit within a unit cube centered at origin.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Normalized mesh
        """
        # Center the mesh
        mesh.vertices = mesh.vertices - mesh.centroid
        
        # Scale to fit in unit cube
        scale = 1.0 / max(mesh.extents)
        mesh.vertices = mesh.vertices * scale
        
        return mesh
    
    def calculate_camera_positions(self, mesh: trimesh.Trimesh) -> List[Tuple[float, float, float]]:
        """
        Calculate camera positions for different viewpoints.
        
        Args:
            mesh: The 3D model mesh
            
        Returns:
            List of (x, y, z) camera positions
        """
        # Get model bounds
        bounds = mesh.bounds
        center = mesh.centroid
        size = mesh.extents
        max_size = max(size)
        
        # Calculate camera distance
        camera_distance = max_size * self.distance_factor
        
        positions = []
        
        # Generate azimuth angles (horizontal rotation)
        azimuth_angles = np.linspace(0, 2 * np.pi, self.num_angles, endpoint=False)
        
        # Generate elevation angles
        elevation_angles = np.linspace(
            np.radians(self.elevation_range[0]), 
            np.radians(self.elevation_range[1]), 
            3  # 3 elevation levels
        )
        
        for azimuth in azimuth_angles:
            for elevation in elevation_angles:
                # Convert spherical to Cartesian coordinates
                x = camera_distance * np.cos(elevation) * np.cos(azimuth)
                y = camera_distance * np.cos(elevation) * np.sin(azimuth)
                z = camera_distance * np.sin(elevation)
                
                # Offset by model center
                camera_pos = center + np.array([x, y, z])
                positions.append(tuple(camera_pos))
        
        return positions
    
    def render_model_from_angle(self, 
                               mesh: trimesh.Trimesh, 
                               camera_pos: Tuple[float, float, float],
                               model_name: str,
                               angle_idx: int) -> Optional[np.ndarray]:
        """
        Render the 3D model from a specific camera angle.
        
        Args:
            mesh: The 3D model mesh
            camera_pos: Camera position (x, y, z)
            model_name: Name of the model (for logging)
            angle_idx: Index of the angle (for logging)
            
        Returns:
            Rendered image as numpy array or None if rendering fails
        """
        try:
            # Set matplotlib backend to Agg for headless rendering
            import matplotlib
            matplotlib.use('Agg')
            
            # Create figure and 3D axis
            fig = plt.figure(figsize=(self.image_size[0]/100, self.image_size[1]/100), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract vertices and faces
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Plot the mesh
            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                           triangles=faces, alpha=0.9, color='lightblue', 
                           edgecolor='black', linewidth=0.1)
            
            # Set axis properties
            ax.set_xlim([mesh.bounds[0][0] - 0.5, mesh.bounds[1][0] + 0.5])
            ax.set_ylim([mesh.bounds[0][1] - 0.5, mesh.bounds[1][1] + 0.5])
            ax.set_zlim([mesh.bounds[0][2] - 0.5, mesh.bounds[1][2] + 0.5])
            
            # Hide axes for cleaner image
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_facecolor('white')
            
            # Calculate view direction
            view_direction = mesh.centroid - np.array(camera_pos)
            view_direction = view_direction / np.linalg.norm(view_direction)
            
            # Set the viewing angle
            azimuth = np.degrees(np.arctan2(view_direction[1], view_direction[0]))
            elevation = np.degrees(np.arcsin(view_direction[2]))
            
            ax.view_init(elev=elevation, azim=azimuth)
            
            # Remove margins
            plt.tight_layout()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            # Convert to numpy array using Agg backend
            fig.canvas.draw()
            
            # Get the buffer and convert to numpy array
            buf = fig.canvas.buffer_rgba()
            image_array = np.asarray(buf)
            
            # Convert RGBA to RGB
            image_array = image_array[:, :, :3]
            
            plt.close(fig)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Failed to render model {model_name} from angle {angle_idx}: {str(e)}")
            try:
                plt.close(fig)
            except:
                pass
            return None
    
    def save_image(self, image_array: np.ndarray, model_name: str, angle_idx: int) -> bool:
        """
        Save the rendered image to the appropriate directory.
        
        Args:
            image_array: Image data as numpy array
            model_name: Name of the model
            angle_idx: Index of the angle
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create model-specific directory
            model_dir = self.output_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_array)
            
            # Resize to desired size
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
            # Save image
            image_path = model_dir / f"{model_name}_angle_{angle_idx:03d}.png"
            image.save(image_path)
            
            logger.info(f"Saved image: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save image for model {model_name}, angle {angle_idx}: {str(e)}")
            return False
    
    def capture_model_frames(self, model_path: str) -> bool:
        """
        Capture frames of a 3D model from different angles.
        
        Args:
            model_path: Path to the 3D model file
            
        Returns:
            True if successful, False otherwise
        """
        # Extract model name from path
        model_name = Path(model_path).parent.name
        
        logger.info(f"Starting frame capture for model: {model_name}")
        
        # Load the 3D model
        mesh = self.load_3d_model(model_path)
        if mesh is None:
            return False
        
        # Normalize the model
        mesh = self.normalize_model(mesh)
        
        # Calculate camera positions
        camera_positions = self.calculate_camera_positions(mesh)
        
        logger.info(f"Capturing {len(camera_positions)} frames for model: {model_name}")
        
        success_count = 0
        for i, camera_pos in enumerate(camera_positions):
            # Render from this angle
            image_array = self.render_model_from_angle(mesh, camera_pos, model_name, i)
            
            if image_array is not None:
                # Save the image
                if self.save_image(image_array, model_name, i):
                    success_count += 1
        
        logger.info(f"Successfully captured {success_count}/{len(camera_positions)} frames for {model_name}")
        return success_count > 0
    
    def process_all_models(self, models_dir: str = "./training_dataset/3dModel") -> None:
        """
        Process all 3D models in the specified directory.
        
        Args:
            models_dir: Directory containing 3D model subdirectories
        """
        models_path = Path(models_dir)
        
        if not models_path.exists():
            logger.error(f"Models directory does not exist: {models_dir}")
            return
        
        # Find all model directories
        model_dirs = [d for d in models_path.iterdir() if d.is_dir()]
        
        logger.info(f"Found {len(model_dirs)} model directories")
        
        for model_dir in model_dirs:
            # Look for .obj files in the directory
            obj_files = list(model_dir.glob("*.obj"))
            
            if obj_files:
                # Use the first .obj file found
                obj_file = obj_files[0]
                logger.info(f"Processing model: {model_dir.name} ({obj_file.name})")
                
                success = self.capture_model_frames(str(obj_file))
                if success:
                    logger.info(f"✓ Successfully processed {model_dir.name}")
                else:
                    logger.error(f"✗ Failed to process {model_dir.name}")
            else:
                logger.warning(f"No .obj files found in {model_dir.name}")


def main():
    """
    Main function to run the frame capture process.
    """
    # Initialize the frame capture utility
    capture_util = ModelFrameCapture(
        output_dir="./training_dataset/2dImg",
        image_size=(512, 512),
        num_angles=12,
        elevation_range=(10, 80),
        distance_factor=2.0
    )
    
    # Process all models
    capture_util.process_all_models("./training_dataset/3dModel")


if __name__ == "__main__":
    main()
