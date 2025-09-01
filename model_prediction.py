import trimesh
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import sys
import os
import cv2
from scipy.spatial import ConvexHull
from cnnModel_pytorch import EnhancedCNNModel
from utils import load_obj, get_device_info

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataPreprocess import dataPreprocess

class EnhancedPredModel:
    """
    Enhanced prediction model with attention mechanisms for complex image understanding.
    """
    def __init__(self, input_img_path, output_mdl_path, model_path, 
                 attention_heads=8, d_model=512):
        self.input_img_path = input_img_path
        self.output_mdl_path = output_mdl_path
        self.model_path = model_path
        
        # Get device info
        device_info = get_device_info()
        self.device = torch.device(device_info['device'])
        print(f"Using device: {self.device}")
        
        # Load the enhanced PyTorch model
        self.model = self.load_enhanced_model(model_path, attention_heads, d_model)
        
        # Enhanced image transformations for complex images
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Image preprocessing for complex backgrounds
        self.preprocessor = ComplexImagePreprocessor()
        
    def load_enhanced_model(self, model_path, attention_heads, d_model):
        """
        Load the enhanced attention-based model.
        """
        try:
            # Get sample data to determine model architecture
            sample_img = Image.open('./shapes2d/square_0.png').convert('RGB').resize((128, 128))
            sample_vertices = load_obj('./shapes3d/cube_0.obj')
            
            # Create enhanced model with same architecture
            model = EnhancedCNNModel(
                128, 128, 3, len(sample_vertices),
                attention_heads=attention_heads, d_model=d_model
            )
            
            # Load trained weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            print(f"Enhanced attention model loaded successfully from {model_path}")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            return model
            
        except Exception as e:
            print(f"Error loading enhanced model: {e}")
            return None
    
    def preprocess_complex_image(self, image_path):
        """
        Preprocess complex images with attention to important features.
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply advanced preprocessing
        processed_image = self.preprocessor.process(image)
        
        # Convert to PIL Image for transforms
        pil_image = Image.fromarray(processed_image)
        
        # Apply transformations
        tensor_image = self.transform(pil_image)
        
        return tensor_image.unsqueeze(0).to(self.device)
    
    def predict(self):
        """
        Generate 3D model from complex 2D image using attention mechanisms.
        """
        if self.model is None:
            print("Error: Enhanced model not loaded")
            return
        
        try:
            # Preprocess complex image
            print("Processing complex image with attention mechanisms...")
            input_tensor = self.preprocess_complex_image(self.input_img_path)
            
            # Make prediction with attention
            with torch.no_grad():
                predicted_vertices = self.model(input_tensor)
                predicted_vertices = predicted_vertices.cpu().numpy()
            
            # Reshape to (N, 3) format
            predicted_vertices = np.reshape(predicted_vertices, (-1, 3))
            
            print(f"Generated {len(predicted_vertices)} vertices with attention")
            
            # Create enhanced mesh from predicted vertices
            predicted_mesh = self.create_enhanced_mesh(predicted_vertices)
            
            if predicted_mesh:
                # Save as OBJ with texture information
                trimesh.exchange.export.export_mesh(predicted_mesh, self.output_mdl_path)
                print(f"Enhanced 3D model saved to {self.output_mdl_path}")
                
                # Generate additional visualization
                self.generate_visualization(predicted_vertices)
            else:
                print("Failed to create 3D mesh")
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
    
    def create_enhanced_mesh(self, vertices):
        """
        Create enhanced 3D mesh with better surface reconstruction.
        """
        if len(vertices) < 4:
            print("Error: Not enough vertices to form the mesh.")
            return None
        
        try:
            # Multiple mesh creation strategies
            meshes = []
            
            # Strategy 1: Convex Hull (good for simple shapes)
            try:
                hull = ConvexHull(vertices)
                convex_mesh = trimesh.Trimesh(vertices=vertices, faces=hull.simplices)
                meshes.append(convex_mesh)
                print("✓ Convex Hull mesh created")
            except Exception as e:
                print(f"Convex Hull failed: {e}")
            
            # Strategy 2: Alpha Shape (better for complex surfaces)
            try:
                from scipy.spatial import Delaunay
                tri = Delaunay(vertices[:, :2])  # Use 2D projection
                alpha_mesh = trimesh.Trimesh(vertices=vertices, faces=tri.simplices)
                meshes.append(alpha_mesh)
                print("✓ Delaunay triangulation mesh created")
            except Exception as e:
                print(f"Delaunay triangulation failed: {e}")
            
            # Strategy 3: Ball Pivoting (best for organic shapes)
            try:
                # Create point cloud first
                point_cloud = trimesh.PointCloud(vertices)
                
                # Estimate normals if not present
                if not hasattr(point_cloud, 'normals') or point_cloud.normals is None:
                    point_cloud.estimate_normals()
                
                # Create mesh using ball pivoting
                ball_mesh = trimesh.creation.triangulate_polygon(
                    point_cloud.vertices, 
                    point_cloud.normals
                )
                if ball_mesh is not None:
                    meshes.append(ball_mesh)
                    print("✓ Ball pivoting mesh created")
            except Exception as e:
                print(f"Ball pivoting failed: {e}")
            
            # Choose the best mesh based on quality metrics
            if not meshes:
                print("All mesh creation strategies failed")
                return None
            
            # Select the best mesh (for now, just use the first successful one)
            best_mesh = meshes[0]
            
            # Clean up the mesh
            best_mesh = best_mesh.process(validate=True)
            
            # Add basic material properties
            best_mesh.visual.face_colors = [200, 200, 200, 255]  # Light gray
            
            return best_mesh
            
        except Exception as e:
            print(f"Error creating enhanced mesh: {e}")
            return None
    
    def generate_visualization(self, vertices):
        """
        Generate additional visualizations of the 3D model.
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # 3D scatter plot of vertices
            fig = plt.figure(figsize=(12, 8))
            
            # Front view (X-Y)
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='o')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title('3D Vertex Distribution')
            
            # Top view (X-Y)
            ax2 = fig.add_subplot(222)
            ax2.scatter(vertices[:, 0], vertices[:, 1], c='r', marker='o')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Top View (X-Y)')
            ax2.grid(True)
            
            # Side view (X-Z)
            ax3 = fig.add_subplot(223)
            ax3.scatter(vertices[:, 0], vertices[:, 2], c='g', marker='o')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Z')
            ax3.set_title('Side View (X-Z)')
            ax3.grid(True)
            
            # Front view (Y-Z)
            ax4 = fig.add_subplot(224)
            ax4.scatter(vertices[:, 1], vertices[:, 2], c='b', marker='o')
            ax4.set_xlabel('Y')
            ax4.set_ylabel('Z')
            ax4.set_title('Front View (Y-Z)')
            ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig('./enhanced_3d_visualization.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✓ 3D visualization saved as 'enhanced_3d_visualization.png'")
            
        except Exception as e:
            print(f"Visualization generation failed: {e}")

class ComplexImagePreprocessor:
    """
    Advanced image preprocessing for complex images with attention to important features.
    """
    def __init__(self):
        self.background_removal = BackgroundRemover()
        self.feature_enhancement = FeatureEnhancer()
        self.noise_reduction = NoiseReducer()
    
    def process(self, image):
        """
        Process complex image with multiple enhancement steps.
        """
        # Step 1: Background removal/focus
        focused_image = self.background_removal.focus_main_object(image)
        
        # Step 2: Feature enhancement
        enhanced_image = self.feature_enhancement.enhance(focused_image)
        
        # Step 3: Noise reduction
        clean_image = self.noise_reduction.reduce_noise(enhanced_image)
        
        return clean_image

class BackgroundRemover:
    """
    Remove or blur background to focus on main objects.
    """
    def focus_main_object(self, image):
        """
        Focus on the main object by detecting and enhancing it.
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to detect objects
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (main object)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask for the main object
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # Apply mask to original image
            focused = cv2.bitwise_and(image, image, mask=mask)
            
            # Create a subtle background
            background = np.full_like(image, [240, 240, 240])  # Light gray
            
            # Combine focused object with subtle background
            result = cv2.addWeighted(focused, 0.9, background, 0.1, 0)
            
            return result
        
        return image

class FeatureEnhancer:
    """
    Enhance important features in the image.
    """
    def enhance(self, image):
        """
        Enhance edges and important features.
        """
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Enhance L channel (lightness)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Enhance edges
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced

class NoiseReducer:
    """
    Reduce noise while preserving important features.
    """
    def reduce_noise(self, image):
        """
        Apply bilateral filtering to reduce noise while preserving edges.
        """
        # Bilateral filter preserves edges while smoothing
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        return denoised

def main():
    """
    Main function for enhanced 3D model prediction.
    """
    # Configuration
    input_img_path = './shapes2d/square_0.png'  # Change this to your complex image
    output_mdl_path = './enhanced_predicted_model.obj'
    model_path = './final_models/enhanced_model_final.pth'  # Enhanced model path
    
    # Enhanced model parameters
    attention_heads = 8
    d_model = 512
    
    # Create enhanced predictor
    predictor = EnhancedPredModel(
        input_img_path, output_mdl_path, model_path,
        attention_heads=attention_heads, d_model=d_model
    )
    
    # Generate 3D model
    predictor.predict()

if __name__ == "__main__":
    main()
