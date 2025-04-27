import numpy as np
import warp as wp
import torch
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path

from wp_kernels import render_gaussians
from utils import world_to_view, projection_matrix, load_ply
from config import *

# Initialize Warp
wp.init()

class GaussianPointCloud:
    def __init__(self, num_points=5000):
        """Initialize a Gaussian point cloud with random parameters."""
        # Point positions (x,y,z)
        self.positions = torch.randn(num_points, 3, dtype=TORCH_FLOAT) * 0.5
        
        # Scaling factors (scale_x, scale_y, scale_z)
        self.scales = torch.ones(num_points, 3, dtype=TORCH_FLOAT) * 0.1
        
        # Rotations as 3x3 matrices
        self.rotations = torch.zeros(num_points, 3, 3, dtype=TORCH_FLOAT)
        for i in range(num_points):
            self.rotations[i] = torch.eye(3, dtype=TORCH_FLOAT)
        
        # Opacity values (0-1)
        self.opacities = torch.ones(num_points, 1, dtype=TORCH_FLOAT) * 0.5
        
        # Spherical harmonics coefficients (assuming degree 3, which means 16 coefficients per channel)
        self.shs = torch.zeros(num_points, 16, 3, dtype=TORCH_FLOAT)
        # Initialize the DC component (first SH coefficient)
        self.shs[:, 0, :] = 0.5
        
        # Mark all parameters as requiring gradients for optimization
        self.positions.requires_grad = True
        self.scales.requires_grad = True
        self.opacities.requires_grad = True
        self.shs.requires_grad = True
        
        # Rotation parameters require special handling for optimization
        # We'll use a separate set of parameters and convert to rotation matrices
        self.rotation_params = torch.zeros(num_points, 3, dtype=TORCH_FLOAT)
        self.rotation_params.requires_grad = True
        
    def get_params_for_optimizer(self):
        """Return parameters that should be optimized."""
        return [self.positions, self.scales, self.opacities, self.shs, self.rotation_params]
        
    def update_rotations(self):
        """Update rotation matrices from rotation parameters."""
        # Convert rotation parameters to rotation matrices
        # This is a simplified version; in a real implementation you might use
        # quaternions or other rotation representations
        for i in range(self.rotation_params.shape[0]):
            rx, ry, rz = self.rotation_params[i]
            
            # Create rotation matrices for each axis
            Rx = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(rx), -torch.sin(rx)],
                [0, torch.sin(rx), torch.cos(rx)]
            ], dtype=TORCH_FLOAT)
            
            Ry = torch.tensor([
                [torch.cos(ry), 0, torch.sin(ry)],
                [0, 1, 0],
                [-torch.sin(ry), 0, torch.cos(ry)]
            ], dtype=TORCH_FLOAT)
            
            Rz = torch.tensor([
                [torch.cos(rz), -torch.sin(rz), 0],
                [torch.sin(rz), torch.cos(rz), 0],
                [0, 0, 1]
            ], dtype=TORCH_FLOAT)
            
            # Combine rotations
            self.rotations[i] = Rx @ Ry @ Rz
    
    def to_numpy(self):
        """Convert torch tensors to numpy arrays for rendering."""
        # Update rotations before conversion
        self.update_rotations()
        
        return {
            'positions': self.positions.detach().numpy(),
            'scales': self.scales.detach().numpy(),
            'rotations': self.rotations.detach().numpy(),
            'opacities': self.opacities.detach().numpy(),
            'shs': self.shs.detach().numpy()
        }
    
    def save_ply(self, filepath):
        """Save the current point cloud to a PLY file."""
        from plyfile import PlyData, PlyElement
        
        # Convert to numpy for export
        data = self.to_numpy()
        
        # Create a structured array for the PLY file
        n = data['positions'].shape[0]
        vertex_data = []
        
        # Basic properties: position, scale, opacity
        for i in range(n):
            vertex = (
                data['positions'][i, 0], data['positions'][i, 1], data['positions'][i, 2],
                data['scales'][i, 0], data['scales'][i, 1], data['scales'][i, 2],
                data['opacities'][i, 0]
            )
            
            # Add rotation matrix elements
            rot_elements = tuple(data['rotations'][i].flatten())
            vertex += rot_elements
            
            # Add SH coefficients
            sh_dc = tuple(data['shs'][i, 0])  # DC component
            vertex += sh_dc
            
            # Add remaining SH coefficients as "f_rest_X" fields
            sh_rest = []
            for j in range(1, 16):  # Skip the DC component (j=0)
                for c in range(3):  # RGB channels
                    sh_rest.append(data['shs'][i, j, c])
            vertex += tuple(sh_rest)
            
            vertex_data.append(vertex)
        
        # Define the structure of the PLY file
        vertex_type = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('opacity', 'f4')
        ]
        
        # Add rotation matrix elements
        for i in range(9):
            vertex_type.append((f'rot_{i}', 'f4'))
        
        # Add SH coefficients
        vertex_type.extend([('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')])
        
        # Add remaining SH coefficients
        for i in range(45):  # 15 coeffs * 3 channels
            vertex_type.append((f'f_rest_{i}', 'f4'))
        
        vertex_array = np.array(vertex_data, dtype=vertex_type)
        el = PlyElement.describe(vertex_array, 'vertex')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the PLY file
        PlyData([el], text=False).write(filepath)


class GaussianSplattingTrainer:
    def __init__(self, dataset_path, output_path, config=None):
        """Initialize the 3D Gaussian Splatting trainer."""
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize default configuration
        self.config = {
            'num_iterations': 30000,
            'learning_rate': 0.01,
            'num_points': 5000,
            'point_batch_size': 1000,
            'densification_interval': 100,
            'pruning_interval': 100,
            'scale_modifier': 1.0,
            'sh_degree': 3,
            'background_color': [0.0, 0.0, 0.0],
            'save_interval': 1000,
        }
        
        # Update with user-provided config
        if config is not None:
            self.config.update(config)
        
        # Load cameras and images
        self.cameras = self.load_cameras()
        self.image_paths = self.load_image_paths()
        print(f"Loaded {len(self.cameras)} cameras and {len(self.image_paths)} images")
        
        # Initialize Gaussian point cloud
        self.point_cloud = GaussianPointCloud(num_points=self.config['num_points'])
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.point_cloud.get_params_for_optimizer(),
            lr=self.config['learning_rate']
        )
        
        # For tracking loss
        self.losses = []
    
    def load_cameras(self):
        """Load camera parameters from cameras.json file."""
        camera_file = self.dataset_path / "cameras.json"
        if not camera_file.exists():
            raise FileNotFoundError(f"No cameras.json found in {self.dataset_path}")
        
        with open(camera_file, 'r') as f:
            camera_data = json.load(f)
        
        cameras = []
        for cam in camera_data:
            # Extract camera parameters
            position = np.array(cam["position"], dtype=np.float32)
            rotation = np.array(cam["rotation"], dtype=np.float32)
            width = cam["width"]
            height = cam["height"]
            fx = cam["fx"]
            fy = cam["fy"]
            
            # Calculate field of view from focal length
            fovx = 2 * np.arctan(width / (2 * fx))
            fovy = 2 * np.arctan(height / (2 * fy))
            
            # Create view matrix
            view_matrix = world_to_view(R=rotation, t=position)
            
            # Create projection matrix
            znear = 0.01
            zfar = 100.0
            proj_matrix = projection_matrix(fovx=fovx, fovy=fovy, znear=znear, zfar=zfar)
            
            # Calculate other parameters
            tan_fovx = np.tan(fovx * 0.5)
            tan_fovy = np.tan(fovy * 0.5)
            
            camera = {
                'id': cam["id"],
                'camera_pos': position,
                'R': rotation,
                'view_matrix': view_matrix,
                'proj_matrix': proj_matrix,
                'tan_fovx': tan_fovx,
                'tan_fovy': tan_fovy,
                'focal_x': fx,
                'focal_y': fy,
                'width': width,
                'height': height
            }
            
            cameras.append(camera)
        
        return cameras
    
    def load_image_paths(self):
        """Load paths to training images."""
        image_dir = self.dataset_path / "images"
        if not image_dir.exists():
            raise FileNotFoundError(f"No images directory found in {self.dataset_path}")
        
        image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        
        return sorted(image_paths)
    
    def load_image(self, path):
        """Load an image as a numpy array."""
        from PIL import Image
        img = Image.open(path)
        img_np = np.array(img) / 255.0  # Normalize to [0, 1]
        return img_np.astype(np.float32)
    
    def render_view(self, camera_idx):
        """Render a view from a specific camera using the current point cloud."""
        camera = self.cameras[camera_idx]
        
        # Get point cloud data as numpy arrays
        data = self.point_cloud.to_numpy()
        
        # Render using the warp renderer
        image = render_gaussians(
            background=np.array(self.config['background_color'], dtype=np.float32),
            means3D=data['positions'],
            colors=None,  # Use SH coefficients instead
            opacity=data['opacities'],
            scales=data['scales'],
            rotations=data['rotations'],
            scale_modifier=self.config['scale_modifier'],
            viewmatrix=camera['view_matrix'],
            projmatrix=camera['proj_matrix'],
            tan_fovx=camera['tan_fovx'],
            tan_fovy=camera['tan_fovy'],
            image_height=camera['height'],
            image_width=camera['width'],
            sh=data['shs'].reshape(-1, 3),  # Reshape to match expected format
            degree=self.config['sh_degree'],
            campos=camera['camera_pos'],
            prefiltered=False,
            antialiasing=True,
            clamped=True
        )
        
        return image
    
    def compute_loss(self, rendered_image, target_image):
        """Compute L2 loss between rendered and target images."""
        # Convert to torch tensors if they're not already
        if not isinstance(rendered_image, torch.Tensor):
            rendered_image = torch.tensor(rendered_image, dtype=TORCH_FLOAT)
        if not isinstance(target_image, torch.Tensor):
            target_image = torch.tensor(target_image, dtype=TORCH_FLOAT)
        
        # Compute mean squared error
        loss = torch.mean((rendered_image - target_image) ** 2)
        return loss
    
    def densification_and_pruning(self, iteration):
        """Perform densification and pruning of Gaussians."""
        # This is a simplified placeholder
        # In a full implementation, you would:
        # 1. Clone Gaussians with high gradient magnitude
        # 2. Remove Gaussians with low opacity or other criteria
        
        # For now, we'll just add a small random perturbation to positions
        if iteration % self.config['densification_interval'] == 0:
            print(f"Iteration {iteration}: Performing densification")
            with torch.no_grad():
                # Add small random perturbation to positions
                noise = torch.randn_like(self.point_cloud.positions) * 0.01
                self.point_cloud.positions += noise
        
        # Similarly simplified pruning logic
        if iteration % self.config['pruning_interval'] == 0:
            print(f"Iteration {iteration}: Performing pruning")
            with torch.no_grad():
                # Keep Gaussians with opacity above threshold
                mask = self.point_cloud.opacities.squeeze() > 0.1
                if mask.sum() > 1000:  # Don't prune below a minimum count
                    self.point_cloud.positions = self.point_cloud.positions[mask]
                    self.point_cloud.scales = self.point_cloud.scales[mask]
                    self.point_cloud.rotations = self.point_cloud.rotations[mask]
                    self.point_cloud.opacities = self.point_cloud.opacities[mask]
                    self.point_cloud.shs = self.point_cloud.shs[mask]
                    self.point_cloud.rotation_params = self.point_cloud.rotation_params[mask]
    
    def save_checkpoint(self, iteration):
        """Save the current point cloud and training state."""
        checkpoint_dir = self.output_path / "point_cloud" / f"iteration_{iteration}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save point cloud as PLY
        ply_path = checkpoint_dir / "point_cloud.ply"
        self.point_cloud.save_ply(ply_path)
        
        # Save loss history
        loss_path = self.output_path / "loss.txt"
        with open(loss_path, 'w') as f:
            for loss in self.losses:
                f.write(f"{loss}\n")
        
        # Save loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig(self.output_path / "loss_plot.png")
        plt.close()
    
    def train(self):
        """Train the 3D Gaussian Splatting model."""
        num_iterations = self.config['num_iterations']
        
        # Main training loop
        with tqdm(total=num_iterations) as pbar:
            for iteration in range(num_iterations):
                # Select a random camera and corresponding image
                camera_idx = np.random.randint(0, len(self.cameras))
                image_path = self.image_paths[camera_idx]
                target_image = self.load_image(image_path)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Render the view
                rendered_image = self.render_view(camera_idx)
                
                # Compute loss
                loss = self.compute_loss(rendered_image, target_image)
                self.losses.append(loss.item())
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                self.optimizer.step()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Loss: {loss.item():.6f}")
                
                # Perform densification and pruning
                self.densification_and_pruning(iteration)
                
                # Save checkpoint
                if iteration % self.config['save_interval'] == 0 or iteration == num_iterations - 1:
                    self.save_checkpoint(iteration)
        
        print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--iterations", type=int, default=30000, help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_points", type=int, default=5000, help="Initial number of Gaussian points")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N iterations")
    
    args = parser.parse_args()
    
    # Configure training
    config = {
        'num_iterations': args.iterations,
        'learning_rate': args.learning_rate,
        'num_points': args.num_points,
        'save_interval': args.save_interval,
    }
    
    # Create trainer and start training
    trainer = GaussianSplattingTrainer(
        dataset_path=args.dataset,
        output_path=args.output,
        config=config
    )
    
    trainer.train()


if __name__ == "__main__":
    main() 