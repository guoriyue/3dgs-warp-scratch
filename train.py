import os
import numpy as np
import matplotlib.pyplot as plt
import warp as wp
import imageio
import json
from tqdm import tqdm
from pathlib import Path
import argparse
from plyfile import PlyData, PlyElement
import math

# Import the renderer and constants
from forward import render_gaussians
from backward import backward, compute_image_loss, backprop_pixel_gradients, densify_gaussians, prune_gaussians, adam_update
from config import *
from utils import *
# Initialize Warp
wp.init()


# Kernels for parameter updates
@wp.kernel
def init_gaussian_params(
    positions: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.vec4),
    opacities: wp.array(dtype=float),
    shs: wp.array(dtype=wp.vec3),
    num_points: int,
    init_scale: float
):
    i = wp.tid()
    if i >= num_points:
        return
    
    # Initialize positions with random values
    # Generate random positions using a single seed derived from the index
    seed = wp.uint32(i)
    positions[i] = wp.vec3(
        wp.randn(seed) * 0.5,
        wp.randn(seed + wp.uint32(1000)) * 0.5,
        wp.randn(seed + wp.uint32(2000)) * 0.5
    )
    
    # Initialize scales
    scales[i] = wp.vec3(init_scale, init_scale, init_scale)
    
    # Initialize rotations to identity matrix
    rotations[i] = wp.vec4(0.0, 0.0, 0.0, 1.0)
    
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
    rot_grad: wp.array(dtype=wp.vec4),
    opacity_grad: wp.array(dtype=float),
    sh_grad: wp.array(dtype=wp.vec3),
    num_points: int
):
    i = wp.tid()
    if i >= num_points:
        return
    
    pos_grad[i] = wp.vec3(0.0, 0.0, 0.0)
    scale_grad[i] = wp.vec3(0.0, 0.0, 0.0)
    rot_grad[i] = wp.vec4(0.0, 0.0, 0.0, 0.0)
    opacity_grad[i] = 0.0
    
    # Zero SH gradients
    for j in range(16):
        idx = i * 16 + j
        sh_grad[idx] = wp.vec3(0.0, 0.0, 0.0)



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
    # NOTE: This function performs compaction on the CPU via numpy, which can be inefficient.
    # Copying data between GPU (Warp) and CPU (NumPy) introduces overhead.
    # A more performant approach would implement compaction directly on the GPU,
    # possibly using parallel prefix sums (scan) to determine new indices.
    mask_np = valid_mask.numpy()
    indices = np.where(mask_np)[0]
    num_valid = len(indices)
    
    # Create new arrays
    new_positions = wp.zeros(num_valid, dtype=wp.vec3)
    new_scales = wp.zeros(num_valid, dtype=wp.vec3)
    new_rotations = wp.zeros(num_valid, dtype=wp.vec4)
    new_opacities = wp.zeros(num_valid, dtype=float)
    new_shs = wp.zeros(num_valid * 16, dtype=wp.vec3)
    
    # Copy valid points
    positions_np = params['positions'].numpy()
    scales_np = params['scales'].numpy()
    rotations_np = params['rotations'].numpy()
    opacities_np = params['opacities'].numpy()
    shs_np = params['shs'].numpy()
    
    for i, idx in enumerate(indices):
        new_positions.numpy()[i] = positions_np[idx]
        new_scales.numpy()[i] = scales_np[idx]
        new_rotations.numpy()[i] = rotations_np[idx]
        new_opacities.numpy()[i] = opacities_np[idx]
        
        # Copy SH coefficients
        for j in range(16):
            new_shs.numpy()[i * 16 + j] = shs_np[idx * 16 + j]
    
    # Create new parameters
    new_params = {
        'positions': new_positions,
        'scales': new_scales,
        'rotations': new_rotations,
        'opacities': new_opacities,
        'shs': new_shs
    }
    
    return new_params, num_valid

# Function to save point cloud to PLY file
def save_ply(params, filepath, num_points):
    # Get numpy arrays
    positions = params['positions'].numpy()
    scales = params['scales'].numpy()
    rotations = params['rotations'].numpy()
    opacities = params['opacities'].numpy()
    shs = params['shs'].numpy()
    
    # Create vertex data
    vertex_data = []
    for i in range(num_points):
        # Basic properties
        vertex = (
            positions[i][0], positions[i][1], positions[i][2],
            scales[i][0], scales[i][1], scales[i][2],
            opacities[i]
        )
        
        # Add rotation quaternion elements
        quat = rotations[i]
        rot_elements = (quat[0], quat[1], quat[2], quat[3])  # x, y, z, w
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
    
    # Add rotation quaternion elements
    vertex_type.extend([('rot_x', 'f4'), ('rot_y', 'f4'), ('rot_z', 'f4'), ('rot_w', 'f4')])
    
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

class NeRFGaussianSplattingTrainer:
    def __init__(self, dataset_path, output_path, config=None):
        """Initialize the 3D Gaussian Splatting trainer using pure Warp for NeRF dataset."""
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize default configuration
        self.config = {
            # 'num_iterations': 30000,
            'num_iterations': 1000,
            'learning_rate': 0.01,
            'num_points': 5000,
            'densification_interval': 100,
            'pruning_interval': 100,
            'scale_modifier': 1.0,
            'sh_degree': 3,
            'background_color': [1.0, 1.0, 1.0],  # White background for NeRF synthetic
            'save_interval': 1000,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_epsilon': 1e-8,
            'initial_scale': 0.1,
            'scene_scale': 1.0,  # Scale factor for the scene
            'camera_angle_x': 0.0,  # Will be loaded from transforms_train.json
            'width': 800,
            'height': 800,
            'near': 2.0,
            'far': 6.0
        }
        
        # Update with user-provided config
        if config is not None:
            self.config.update(config)
        
        # Load NeRF dataset
        print(f"Loading NeRF dataset from {self.dataset_path}")
        self.cameras, self.image_paths = self.load_nerf_data()
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
        rotations = wp.zeros(self.num_points, dtype=wp.vec4)
        opacities = wp.zeros(self.num_points, dtype=float)
        shs = wp.zeros(self.num_points * 16, dtype=wp.vec3)  # 16 coeffs per point
        
        # Launch kernel to initialize parameters
        wp.launch(
            init_gaussian_params,
            dim=self.num_points,
            inputs=[positions, scales, rotations, opacities, shs, self.num_points, self.config['initial_scale']]
        )
        
        # Return parameters as dictionary
        return {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'opacities': opacities,
            'shs': shs
        }
    
    def create_gradient_arrays(self):
        """Create arrays for gradients or optimizer state."""
        positions = wp.zeros(self.num_points, dtype=wp.vec3)
        scales = wp.zeros(self.num_points, dtype=wp.vec3)
        rotations = wp.zeros(self.num_points, dtype=wp.vec4)
        opacities = wp.zeros(self.num_points, dtype=float)
        shs = wp.zeros(self.num_points * 16, dtype=wp.vec3)
        
        # Return a dictionary of arrays
        return {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'opacities': opacities,
            'shs': shs
        }
    
    def load_nerf_data(self):
        """Load camera parameters and images from a NeRF dataset."""
        # Read transforms_train.json
        transforms_path = self.dataset_path / "transforms_train.json"
        if not transforms_path.exists():
            raise FileNotFoundError(f"No transforms_train.json found in {self.dataset_path}")
        
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        
        # Extract global parameters from the transforms file
        self.config['camera_angle_x'] = transforms.get('camera_angle_x', 0.6911112070083618)  # Default from lego
        width = height = 800  # Default for NeRF synthetic
        
        # Calculate focal length
        focal = 0.5 * width / np.tan(0.5 * self.config['camera_angle_x'])
        
        cameras = []
        image_paths = []
        
        # Process each frame
        for i, frame in enumerate(transforms['frames']):
            # Extract camera-to-world transform matrix
            cam2world = np.array(frame['transform_matrix'], dtype=np.float32)
            
            # Extract camera position (translation part of cam2world)
            position = cam2world[:3, 3]
            
            # Extract rotation (rotation part of cam2world)
            rotation = cam2world[:3, :3]
            
            # Get image path
            img_path = str(self.dataset_path / f"{frame['file_path']}.png")
            image_paths.append(img_path)
            
            # For our renderer, we need world-to-camera transform
            world2cam = np.zeros((4, 4), dtype=np.float32)
            world2cam[:3, :3] = rotation.T  # Transpose for inverse rotation
            world2cam[:3, 3] = -rotation.T @ position  # -R^T * t for translation
            world2cam[3, 3] = 1.0
            
            # Create view matrix (same as world2cam for our purposes)
            view_matrix = world2cam
            
            # Calculate fov from focal length
            fovx = 2 * np.arctan(width / (2 * focal))
            fovy = 2 * np.arctan(height / (2 * focal))
            
            # Create projection matrix
            znear = self.config['near']
            zfar = self.config['far']
            proj_matrix = projection_matrix(fovx=fovx, fovy=fovy, znear=znear, zfar=zfar)
            
            # Calculate other parameters
            tan_fovx = np.tan(fovx * 0.5)
            tan_fovy = np.tan(fovy * 0.5)
            
            camera = {
                'id': i,
                'camera_pos': position,
                'R': rotation,
                'view_matrix': view_matrix,
                'proj_matrix': proj_matrix,
                'tan_fovx': tan_fovx,
                'tan_fovy': tan_fovy,
                'focal_x': focal,
                'focal_y': focal,
                'width': width,
                'height': height
            }
            
            cameras.append(camera)
        
        return cameras, image_paths
    
    def load_image(self, path):
        """Load an image as a numpy array."""
        if os.path.exists(path):
            img = imageio.imread(path)
            # Convert to float and normalize to [0, 1]
            img_np = img.astype(np.float32) / 255.0
            # Ensure image is RGB (discard alpha channel if present)
            if img_np.shape[2] == 4:
                img_np = img_np[:, :, :3] # Keep only R, G, B channels
            return img_np
        else:
            raise FileNotFoundError(f"Image not found: {path}")
    
    def render_view(self, camera_idx):
        """Render a view from a specific camera using the current point cloud."""
        # Note: `render_gaussians` is now imported at the top of the file.
        camera = self.cameras[camera_idx]
        
        # Get point cloud data as numpy arrays
        positions_np = self.params['positions'].numpy()
        scales_np = self.params['scales'].numpy()
        rotations_np = self.params['rotations'].numpy()
        opacities_np = self.params['opacities'].numpy()
        shs_np = self.params['shs'].numpy()
        
        # Render using the warp renderer
        rendered_img, depth_image = render_gaussians(
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
        
        return rendered_img
    
    def get_render_buffers(self):
        """Return the last set of buffers from the rendering process.
        This function should be called after render_view.
        
        Returns:
            Tuple of buffer arrays needed for backward pass
        """
        # Unfortunately, render_gaussians doesn't expose these buffers
        # We need to modify the rendering code to save these arrays
        
        # For now, create mock buffers with appropriate sizes
        num_points = self.params['positions'].shape[0]
        
        # Create placeholder buffers
        radii = wp.zeros(num_points, dtype=int, device=DEVICE)
        points_xy_image = wp.zeros(num_points, dtype=wp.vec2, device=DEVICE)
        depths = wp.zeros(num_points, dtype=float, device=DEVICE)
        rgb = wp.zeros(num_points, dtype=wp.vec3, device=DEVICE)
        conic_opacity = wp.zeros(num_points, dtype=wp.vec4, device=DEVICE)
        
        # Number of rendered points (we don't know this without running forward pass)
        # This is a very rough estimate - all points visible from all tiles
        image_width = self.cameras[0]['width']
        image_height = self.cameras[0]['height']
        max_rendered = num_points * 10  # Overestimate
        
        point_list = wp.zeros(max_rendered, dtype=int, device=DEVICE)
        
        # Calculate tile grid for spatial optimization
        TILE_M = 16  # These should match the values in forward.py
        TILE_N = 16
        tile_grid_x = (image_width + TILE_M - 1) // TILE_M
        tile_grid_y = (image_height + TILE_N - 1) // TILE_N
        tile_count = tile_grid_x * tile_grid_y
        
        ranges = wp.zeros(tile_count, dtype=wp.vec2i, device=DEVICE)
                
        return radii, points_xy_image, depths, rgb, conic_opacity, point_list, ranges
    
    def zero_grad(self):
        """Zero out all gradients."""
        wp.launch(
            zero_gradients,
            dim=self.num_points,
            inputs=[
                self.grads['positions'],
                self.grads['scales'],
                self.grads['rotations'],
                self.grads['opacities'],
                self.grads['shs'],
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
                    self.params['positions'],
                    self.grads['positions'],
                    self.grads['scales'],
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
                    self.params['opacities'],
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
                # Parameters
                self.params['positions'],
                self.params['scales'],
                self.params['rotations'],
                self.params['opacities'],
                self.params['shs'],
                
                # Gradients
                self.grads['positions'],
                self.grads['scales'],
                self.grads['rotations'],
                self.grads['opacities'],
                self.grads['shs'],
                
                # First moments (m)
                self.adam_m['positions'],
                self.adam_m['scales'],
                self.adam_m['rotations'],
                self.adam_m['opacities'],
                self.adam_m['shs'],
                
                # Second moments (v)
                self.adam_v['positions'],
                self.adam_v['scales'],
                self.adam_v['rotations'],
                self.adam_v['opacities'],
                self.adam_v['shs'],
                
                # Optimizer parameters
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
        
        # Save a rendered view
        camera_idx = 0  # Front view
        rendered_image = self.render_view(camera_idx)
        plt.figure(figsize=(10, 10))
        plt.imshow(rendered_image)
        plt.title(f'Rendered View at Iteration {iteration}')
        plt.axis('off')
        plt.savefig(checkpoint_dir / "rendered_view.png")
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

                # --- CRITICAL: Missing Gradient Computation ---
                # The core backward pass is missing here.
                # We have pixel_grad (dL/dColor), but need dL/dParams (positions, scales, etc.).
                # This requires a differentiable renderer or a manually implemented backward pass
                # for the `render_gaussians` function from `wp_kernels`.
                # The backward pass would compute how changes in each Gaussian parameter
                # affect the final pixel colors and propagate the pixel gradients back.

                # 1. Compute pixel gradients (dL/dColor)
                height, width = rendered_image.shape[0], rendered_image.shape[1]
                pixel_grad_buffer = wp.zeros((height, width), dtype=wp.vec3, device=DEVICE)
                wp.launch(
                    kernel=backprop_pixel_gradients,
                    dim=(width, height),
                    inputs=[
                        rendered_image,
                        target_image,
                        pixel_grad_buffer,
                        width,
                        height
                    ]
                )
                
                
                # Prepare camera parameters
                camera = self.cameras[camera_idx]
                view_matrix = wp.mat44(camera['view_matrix'].flatten())
                proj_matrix = wp.mat44(camera['proj_matrix'].flatten())
                campos = wp.vec3(camera['camera_pos'][0], camera['camera_pos'][1], camera['camera_pos'][2])
                
                # For the render_gaussians function in forward.py, we need to collect all the 
                # intermediate data that was computed during the forward pass
                
                # Get these values from the last call to render_gaussians
                # We need to modify render_view to return these buffers
                radii, points_xy_image, depths, rgb, conic_opacity, point_list, ranges = \
                    self.get_render_buffers()
                    
                # Create appropriate buffer dictionaries for the backward pass
                geom_buffer = {
                    'radii': radii,
                    'means2D': points_xy_image,
                    'conic_opacity': conic_opacity,
                    'rgb': rgb
                }
                
                binning_buffer = {
                    'point_list': point_list
                }
                
                # To get final_Ts and n_contrib, we would need to save these from the forward pass
                # For now, create empty placeholders
                final_Ts = wp.zeros((height, width), dtype=float, device=DEVICE)
                n_contrib = wp.zeros((height, width), dtype=int, device=DEVICE)
                
                img_buffer = {
                    'ranges': ranges,
                    'final_Ts': final_Ts,
                    'n_contrib': n_contrib
                }
                
                gradients = backward(
                    # Core parameters
                    background=np.array(self.config['background_color'], dtype=np.float32),
                    means3D=self.params['positions'],
                    dL_dpixels=pixel_grad_buffer,
                    
                    # Model parameters (pass directly from self.params)
                    opacity=self.params['opacities'],
                    shs=self.params['shs'],
                    scales=self.params['scales'],
                    rotations=self.params['rotations'],
                    scale_modifier=self.config['scale_modifier'],
                    
                    # Camera parameters
                    viewmatrix=view_matrix,
                    projmatrix=proj_matrix,
                    tan_fovx=camera['tan_fovx'],
                    tan_fovy=camera['tan_fovy'],
                    image_height=camera['height'],
                    image_width=camera['width'],
                    campos=campos,
                    
                    # Forward output buffers
                    radii=radii,
                    means2D=points_xy_image,
                    conic_opacity=conic_opacity,
                    rgb=rgb,
                    
                    # Internal state buffers
                    geom_buffer=geom_buffer,
                    binning_buffer=binning_buffer,
                    img_buffer=img_buffer,
                    
                    # Algorithm parameters
                    degree=self.config['sh_degree'],
                    debug=False
                )
                
                # 3. Copy gradients from backward result to the optimizer's gradient buffers
                wp.copy(self.grads['positions'], gradients['dL_dmean3D'])
                wp.copy(self.grads['scales'], gradients['dL_dscale'])
                wp.copy(self.grads['rotations'], gradients['dL_drot'])
                wp.copy(self.grads['opacities'], gradients['dL_dopacity'])
                wp.copy(self.grads['shs'], gradients['dL_dshs'])

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
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting model with NeRF dataset")
    parser.add_argument("--dataset", type=str, default="/Users/guomingfei/Desktop/warp-nerf-scratch/data/nerf_synthetic/lego",
                        help="Path to NeRF dataset directory (default: Lego dataset)")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--iterations", type=int, default=300, help="Number of training iterations")
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
    trainer = NeRFGaussianSplattingTrainer(
        dataset_path=args.dataset,
        output_path=args.output,
        config=config
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
