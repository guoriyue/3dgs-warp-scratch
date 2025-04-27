import numpy as np
import warp as wp
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path
from plyfile import PlyData, PlyElement

from utils import world_to_view, projection_matrix, load_ply
from config import *

# Initialize Warp
wp.init()

# Define structures for Gaussian parameters
@wp.struct
class GaussianParams:
    positions: wp.array(dtype=wp.vec3)
    scales: wp.array(dtype=wp.vec3)
    rotations: wp.array(dtype=wp.mat33)
    opacities: wp.array(dtype=float)
    shs: wp.array(dtype=wp.vec3)  # Flattened SH coefficients

@wp.struct
class TrainingState:
    params: GaussianParams
    grads: GaussianParams
    adam_m: GaussianParams  # Adam optimizer first moment
    adam_v: GaussianParams  # Adam optimizer second moment
    step: int

# Kernels for parameter updates
@wp.kernel
def init_gaussian_params(
    positions: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.mat33),
    opacities: wp.array(dtype=float),
    shs: wp.array(dtype=wp.vec3),
    num_points: int,
    init_scale: float
):
    i = wp.tid()
    if i >= num_points:
        return
    
    # Initialize positions with random values
    positions[i] = wp.vec3(
        wp.randn(0, float(i), 0) * 0.5,
        wp.randn(0, float(i), 1) * 0.5,
        wp.randn(0, float(i), 2) * 0.5
    )
    
    # Initialize scales
    scales[i] = wp.vec3(init_scale, init_scale, init_scale)
    
    # Initialize rotations to identity matrix
    rotations[i] = wp.mat33(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
    
    # Initialize opacities
    opacities[i] = 0.5
    
    # Initialize SH coefficients (just DC term for now)
    for j in range(16):  # Assuming degree 3 (16 coefficients)
        idx = i * 16 + j
        if j == 0:  # DC term
            shs[idx] = wp.vec3(0.5, 0.5, 0.5)
        else:
            shs[idx] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def zero_gradients(
    pos_grad: wp.array(dtype=wp.vec3),
    scale_grad: wp.array(dtype=wp.vec3),
    rot_grad: wp.array(dtype=wp.mat33),
    opacity_grad: wp.array(dtype=float),
    sh_grad: wp.array(dtype=wp.vec3),
    num_points: int
):
    i = wp.tid()
    if i >= num_points:
        return
    
    pos_grad[i] = wp.vec3(0.0, 0.0, 0.0)
    scale_grad[i] = wp.vec3(0.0, 0.0, 0.0)
    rot_grad[i] = wp.mat33(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    )
    opacity_grad[i] = 0.0
    
    # Zero SH gradients
    for j in range(16):
        idx = i * 16 + j
        sh_grad[idx] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def adam_update(
    params: GaussianParams,
    grads: GaussianParams,
    m: GaussianParams,
    v: GaussianParams,
    num_points: int,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    iteration: int
):
    i = wp.tid()
    if i >= num_points:
        return
    
    # Bias correction terms
    bias_correction1 = 1.0 - wp.pow(beta1, float(iteration + 1))
    bias_correction2 = 1.0 - wp.pow(beta2, float(iteration + 1))
    
    # Update positions
    m.positions[i] = beta1 * m.positions[i] + (1.0 - beta1) * grads.positions[i]
    v.positions[i] = beta2 * v.positions[i] + (1.0 - beta2) * grads.positions[i] * grads.positions[i]
    m_corrected = m.positions[i] / bias_correction1
    v_corrected = v.positions[i] / bias_correction2
    params.positions[i] = params.positions[i] - lr * m_corrected / (wp.sqrt(v_corrected) + epsilon)
    
    # Update scales (with some constraints to keep them positive)
    m.scales[i] = beta1 * m.scales[i] + (1.0 - beta1) * grads.scales[i]
    v.scales[i] = beta2 * v.scales[i] + (1.0 - beta2) * grads.scales[i] * grads.scales[i]
    m_corrected = m.scales[i] / bias_correction1
    v_corrected = v.scales[i] / bias_correction2
    scale_update = lr * m_corrected / (wp.sqrt(v_corrected) + epsilon)
    params.scales[i] = wp.vec3(
        wp.max(params.scales[i][0] - scale_update[0], 0.001),
        wp.max(params.scales[i][1] - scale_update[1], 0.001),
        wp.max(params.scales[i][2] - scale_update[2], 0.001)
    )
    
    # Update opacity (with clamping to [0,1])
    m.opacities[i] = beta1 * m.opacities[i] + (1.0 - beta1) * grads.opacities[i]
    v.opacities[i] = beta2 * v.opacities[i] + (1.0 - beta2) * grads.opacities[i] * grads.opacities[i]
    m_corrected = m.opacities[i] / bias_correction1
    v_corrected = v.opacities[i] / bias_correction2
    opacity_update = lr * m_corrected / (wp.sqrt(v_corrected) + epsilon)
    params.opacities[i] = wp.max(wp.min(params.opacities[i] - opacity_update, 1.0), 0.0)
    
    # Update SH coefficients
    for j in range(16):
        idx = i * 16 + j
        m.shs[idx] = beta1 * m.shs[idx] + (1.0 - beta1) * grads.shs[idx]
        v.shs[idx] = beta2 * v.shs[idx] + (1.0 - beta2) * grads.shs[idx] * grads.shs[idx]
        m_corrected = m.shs[idx] / bias_correction1
        v_corrected = v.shs[idx] / bias_correction2
        params.shs[idx] = params.shs[idx] - lr * m_corrected / (wp.sqrt(v_corrected) + epsilon)

@wp.kernel
def compute_image_loss(
    rendered: wp.array2d(dtype=wp.vec3),
    target: wp.array2d(dtype=wp.vec3),
    loss_buffer: wp.array(dtype=float),
    width: int,
    height: int
):
    i, j = wp.tid()
    if i >= width or j >= height:
        return
    
    # Compute squared difference for each pixel component
    rendered_pixel = rendered[i, j]
    target_pixel = target[i, j]
    diff = rendered_pixel - target_pixel
    squared_diff = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
    
    # Atomic add to loss buffer
    wp.atomic_add(loss_buffer, 0, squared_diff)

@wp.kernel
def backprop_pixel_gradients(
    rendered: wp.array2d(dtype=wp.vec3),
    target: wp.array2d(dtype=wp.vec3),
    pixel_grad: wp.array2d(dtype=wp.vec3),
    width: int,
    height: int
):
    i, j = wp.tid()
    if i >= width or j >= height:
        return
    
    # Compute gradient (2 * diff for MSE loss)
    rendered_pixel = rendered[i, j]
    target_pixel = target[i, j]
    diff = rendered_pixel - target_pixel
    
    # 2 * diff for mean squared error gradient
    pixel_grad[i, j] = wp.vec3(
        2.0 * diff[0], 
        2.0 * diff[1], 
        2.0 * diff[2]
    )

@wp.kernel
def densify_gaussians(
    positions: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.mat33),
    opacities: wp.array(dtype=float),
    shs: wp.array(dtype=wp.vec3),
    pos_grads: wp.array(dtype=wp.vec3),
    scale_grads: wp.array(dtype=wp.vec3),
    num_points: int,
    noise_scale: float
):
    i = wp.tid()
    if i >= num_points:
        return
    
    # Get gradient magnitudes to identify high-gradient Gaussians
    pos_grad_mag = wp.length(pos_grads[i])
    scale_grad_mag = wp.length(scale_grads[i])
    
    # We just add noise to positions based on gradients
    # In a full implementation, you'd clone Gaussians with high gradient magnitude
    if pos_grad_mag > 0.1 or scale_grad_mag > 0.1:
        # Add random noise to position
        noise = wp.vec3(
            wp.randn(float(i), 0.0, 1.0) * noise_scale,
            wp.randn(float(i), 0.0, 2.0) * noise_scale,
            wp.randn(float(i), 0.0, 3.0) * noise_scale
        )
        positions[i] = positions[i] + noise

@wp.kernel
def prune_gaussians(
    positions: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.mat33),
    opacities: wp.array(dtype=float),
    shs: wp.array(dtype=wp.vec3),
    opacity_threshold: float,
    valid_mask: wp.array(dtype=int),
    num_points: int
):
    i = wp.tid()
    if i >= num_points:
        return
    
    # Mark Gaussians for keeping or removal
    if opacities[i] > opacity_threshold:
        valid_mask[i] = 1
    else:
        valid_mask[i] = 0

# Function to compute MSE loss
def compute_loss(rendered_img, target_img):
    height, width = rendered_img.shape[0], rendered_img.shape[1]
    
    # Create device arrays
    d_rendered = wp.array(rendered_img, dtype=wp.vec3)
    d_target = wp.array(target_img, dtype=wp.vec3)
    
    # Create loss buffer
    loss_buffer = wp.zeros(1, dtype=float)
    
    # Compute loss
    wp.launch(
        compute_image_loss,
        dim=(width, height),
        inputs=[d_rendered, d_target, loss_buffer, width, height]
    )
    
    # Get loss value
    loss = float(loss_buffer.numpy()[0]) / (width * height)
    return loss

# Function to create a new point cloud with a reduced set of points
def compact_point_cloud(params, valid_mask):
    mask_np = valid_mask.numpy()
    indices = np.where(mask_np)[0]
    num_valid = len(indices)
    
    # Create new arrays
    new_positions = wp.zeros(num_valid, dtype=wp.vec3)
    new_scales = wp.zeros(num_valid, dtype=wp.vec3)
    new_rotations = wp.zeros(num_valid, dtype=wp.mat33)
    new_opacities = wp.zeros(num_valid, dtype=float)
    new_shs = wp.zeros(num_valid * 16, dtype=wp.vec3)
    
    # Copy valid points
    positions_np = params.positions.numpy()
    scales_np = params.scales.numpy()
    rotations_np = params.rotations.numpy()
    opacities_np = params.opacities.numpy()
    shs_np = params.shs.numpy()
    
    for i, idx in enumerate(indices):
        new_positions.numpy()[i] = positions_np[idx]
        new_scales.numpy()[i] = scales_np[idx]
        new_rotations.numpy()[i] = rotations_np[idx]
        new_opacities.numpy()[i] = opacities_np[idx]
        
        # Copy SH coefficients
        for j in range(16):
            new_shs.numpy()[i * 16 + j] = shs_np[idx * 16 + j]
    
    # Create new parameters
    new_params = GaussianParams()
    new_params.positions = new_positions
    new_params.scales = new_scales
    new_params.rotations = new_rotations
    new_params.opacities = new_opacities
    new_params.shs = new_shs
    
    return new_params, num_valid

# Function to save point cloud to PLY file
def save_ply(params, filepath, num_points):
    # Get numpy arrays
    positions = params.positions.numpy()
    scales = params.scales.numpy()
    rotations = params.rotations.numpy()
    opacities = params.opacities.numpy()
    shs = params.shs.numpy()
    
    # Create vertex data
    vertex_data = []
    for i in range(num_points):
        # Basic properties
        vertex = (
            positions[i][0], positions[i][1], positions[i][2],
            scales[i][0], scales[i][1], scales[i][2],
            opacities[i]
        )
        
        # Add rotation matrix elements
        rot_matrix = rotations[i]
        rot_elements = (
            rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2],
            rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2],
            rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2]
        )
        vertex += rot_elements
        
        # Add SH coefficients
        sh_dc = tuple(shs[i * 16][j] for j in range(3))
        vertex += sh_dc
        
        # Add remaining SH coefficients
        sh_rest = []
        for j in range(1, 16):
            for c in range(3):
                sh_rest.append(shs[i * 16 + j][c])
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
    print(f"Point cloud saved to {filepath}")

class WarpGaussianSplattingTrainer:
    def __init__(self, dataset_path, output_path, config=None):
        """Initialize the 3D Gaussian Splatting trainer using pure Warp."""
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize default configuration
        self.config = {
            'num_iterations': 30000,
            'learning_rate': 0.01,
            'num_points': 5000,
            'densification_interval': 100,
            'pruning_interval': 100,
            'scale_modifier': 1.0,
            'sh_degree': 3,
            'background_color': [0.0, 0.0, 0.0],
            'save_interval': 1000,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_epsilon': 1e-8,
            'initial_scale': 0.1
        }
        
        # Update with user-provided config
        if config is not None:
            self.config.update(config)
        
        # Load cameras and images
        self.cameras = self.load_cameras()
        self.image_paths = self.load_image_paths()
        print(f"Loaded {len(self.cameras)} cameras and {len(self.image_paths)} images")
        
        # Initialize parameters
        self.num_points = self.config['num_points']
        self.params = self.initialize_parameters()
        
        # Create gradient arrays
        self.grads = self.create_gradient_arrays()
        
        # Create optimizer state
        self.adam_m = self.create_gradient_arrays()  # First moment
        self.adam_v = self.create_gradient_arrays()  # Second moment
        
        # For tracking loss
        self.losses = []
    
    def initialize_parameters(self):
        """Initialize Gaussian parameters."""
        positions = wp.zeros(self.num_points, dtype=wp.vec3)
        scales = wp.zeros(self.num_points, dtype=wp.vec3)
        rotations = wp.zeros(self.num_points, dtype=wp.mat33)
        opacities = wp.zeros(self.num_points, dtype=float)
        shs = wp.zeros(self.num_points * 16, dtype=wp.vec3)  # 16 coeffs per point
        
        # Launch kernel to initialize parameters
        wp.launch(
            init_gaussian_params,
            dim=self.num_points,
            inputs=[positions, scales, rotations, opacities, shs, self.num_points, self.config['initial_scale']]
        )
        
        # Create parameter struct
        params = GaussianParams()
        params.positions = positions
        params.scales = scales
        params.rotations = rotations
        params.opacities = opacities
        params.shs = shs
        
        return params
    
    def create_gradient_arrays(self):
        """Create arrays for gradients or optimizer state."""
        grads = GaussianParams()
        grads.positions = wp.zeros(self.num_points, dtype=wp.vec3)
        grads.scales = wp.zeros(self.num_points, dtype=wp.vec3)
        grads.rotations = wp.zeros(self.num_points, dtype=wp.mat33)
        grads.opacities = wp.zeros(self.num_points, dtype=float)
        grads.shs = wp.zeros(self.num_points * 16, dtype=wp.vec3)
        
        return grads
    
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
        from wp_kernels import render_gaussians
        
        camera = self.cameras[camera_idx]
        
        # Get point cloud data as numpy arrays
        positions_np = self.params.positions.numpy()
        scales_np = self.params.scales.numpy()
        rotations_np = self.params.rotations.numpy()
        opacities_np = self.params.opacities.numpy()
        shs_np = self.params.shs.numpy()
        
        # Render using the warp renderer
        image = render_gaussians(
            background=np.array(self.config['background_color'], dtype=np.float32),
            means3D=positions_np,
            colors=None,  # Use SH coefficients instead
            opacity=opacities_np,
            scales=scales_np,
            rotations=rotations_np,
            scale_modifier=self.config['scale_modifier'],
            viewmatrix=camera['view_matrix'],
            projmatrix=camera['proj_matrix'],
            tan_fovx=camera['tan_fovx'],
            tan_fovy=camera['tan_fovy'],
            image_height=camera['height'],
            image_width=camera['width'],
            sh=shs_np,  # Pass SH coefficients
            degree=self.config['sh_degree'],
            campos=camera['camera_pos'],
            prefiltered=False,
            antialiasing=True,
            clamped=True
        )
        
        return image
    
    def zero_grad(self):
        """Zero out all gradients."""
        wp.launch(
            zero_gradients,
            dim=self.num_points,
            inputs=[
                self.grads.positions,
                self.grads.scales,
                self.grads.rotations,
                self.grads.opacities,
                self.grads.shs,
                self.num_points
            ]
        )
    
    def densification_and_pruning(self, iteration):
        """Perform densification and pruning of Gaussians."""
        # Simplified densification - add noise to positions based on gradients
        if iteration % self.config['densification_interval'] == 0:
            print(f"Iteration {iteration}: Performing densification")
            wp.launch(
                densify_gaussians,
                dim=self.num_points,
                inputs=[
                    self.params.positions,
                    self.params.scales,
                    self.params.rotations,
                    self.params.opacities,
                    self.params.shs,
                    self.grads.positions,
                    self.grads.scales,
                    self.num_points,
                    0.01  # noise scale
                ]
            )
        
        # Simplified pruning - remove Gaussians with low opacity
        if iteration % self.config['pruning_interval'] == 0:
            print(f"Iteration {iteration}: Performing pruning")
            # Create mask for valid Gaussians
            valid_mask = wp.zeros(self.num_points, dtype=int)
            
            wp.launch(
                prune_gaussians,
                dim=self.num_points,
                inputs=[
                    self.params.positions,
                    self.params.scales,
                    self.params.rotations,
                    self.params.opacities,
                    self.params.shs,
                    0.1,  # opacity threshold
                    valid_mask,
                    self.num_points
                ]
            )
            
            # Count valid points
            valid_count = int(np.sum(valid_mask.numpy()))
            
            # Only prune if we have enough valid points
            if valid_count > 1000 and valid_count < self.num_points:
                print(f"Pruning point cloud from {self.num_points} to {valid_count} points")
                
                # Create new parameters with only valid points
                new_params, new_num_points = compact_point_cloud(self.params, valid_mask)
                
                # Update parameters and count
                self.params = new_params
                self.num_points = new_num_points
                
                # Create new gradient arrays
                self.grads = self.create_gradient_arrays()
                self.adam_m = self.create_gradient_arrays()
                self.adam_v = self.create_gradient_arrays()
    
    def optimizer_step(self, iteration):
        """Perform an Adam optimization step."""
        wp.launch(
            adam_update,
            dim=self.num_points,
            inputs=[
                self.params,
                self.grads,
                self.adam_m,
                self.adam_v,
                self.num_points,
                self.config['learning_rate'],
                self.config['adam_beta1'],
                self.config['adam_beta2'],
                self.config['adam_epsilon'],
                iteration
            ]
        )
    
    def save_checkpoint(self, iteration):
        """Save the current point cloud and training state."""
        checkpoint_dir = self.output_path / "point_cloud" / f"iteration_{iteration}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save point cloud as PLY
        ply_path = checkpoint_dir / "point_cloud.ply"
        save_ply(self.params, ply_path, self.num_points)
        
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
                self.zero_grad()
                
                # Render the view
                rendered_image = self.render_view(camera_idx)
                
                # Compute loss
                loss = compute_loss(rendered_image, target_image)
                self.losses.append(loss)
                
                # Compute gradients (simplified - in a full implementation, you'd do this with proper autograd)
                # For now, we just use a simple approach where we compute pixel gradients 
                # and assume the render_gaussians function propagates them
                # This is a placeholder - in practice, you'd need to implement a proper gradient system
                
                # Update parameters
                self.optimizer_step(iteration)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Loss: {loss:.6f}")
                
                # Perform densification and pruning
                self.densification_and_pruning(iteration)
                
                # Save checkpoint
                if iteration % self.config['save_interval'] == 0 or iteration == num_iterations - 1:
                    self.save_checkpoint(iteration)
        
        print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting model using pure Warp")
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
    trainer = WarpGaussianSplattingTrainer(
        dataset_path=args.dataset,
        output_path=args.output,
        config=config
    )
    
    trainer.train()


if __name__ == "__main__":
    main() 