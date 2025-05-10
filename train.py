import os
import numpy as np
import matplotlib.pyplot as plt
import warp as wp
import imageio
import json
from tqdm import tqdm
from pathlib import Path
import argparse
import cv2

# Import the renderer and constants
from forward import render_gaussians
from backward import backward, densify_gaussians, prune_gaussians, adam_update
from config import *
from utils import *
from loss import l1_loss, ssim, compute_image_gradients, depth_loss
                    
# Initialize Warp
wp.init()

# Kernels for parameter updates
@wp.kernel
def init_gaussian_params(
    camera_center: wp.vec3,
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
    seed = i
    
    offset = wp.vec3(
        wp.randn(wp.uint32(seed)) * 0.7,
        wp.randn(wp.uint32(seed + 1)) * 0.7,
        wp.randn(wp.uint32(seed + 2)) * 0.7,
    )
    positions[i] = camera_center + offset
    
    # Initialize scales
    scales[i] = wp.vec3(init_scale, init_scale, init_scale)
    
    # Initialize rotations to identity matrix
    rotations[i] = wp.vec4(0.0, 0.0, 0.0, 1.0)
    
    # Initialize opacities
    opacities[i] = 0.5
    
    # Initialize SH coefficients (just DC term for now)
    for j in range(16):  # degree=3, total 16 coefficients
        idx = i * 16 + j
        # Slight random initialization with positive bias
        shs[idx] = wp.vec3(
            wp.clamp(wp.randn(wp.uint32(seed + j * 100)) * 0.3 + 0.5, 0.0, 1.0),
            wp.clamp(wp.randn(wp.uint32(seed + j * 100 + 1)) * 0.3 + 0.5, 0.0, 1.0),
            wp.clamp(wp.randn(wp.uint32(seed + j * 100 + 2)) * 0.3 + 0.5, 0.0, 1.0),
        )

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


class NeRFGaussianSplattingTrainer:
    def __init__(self, dataset_path, output_path, config=None):
        """Initialize the 3D Gaussian Splatting trainer using pure Warp for NeRF dataset."""
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize configuration from GaussianParams
        self.config = GaussianParams.get_config_dict()
        
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
        
        # Initialize intermediate buffers dictionary
        self.intermediate_buffers = {}
    
    def initialize_parameters(self):
        """Initialize Gaussian parameters."""
        positions = wp.zeros(self.num_points, dtype=wp.vec3)
        scales = wp.zeros(self.num_points, dtype=wp.vec3)
        rotations = wp.zeros(self.num_points, dtype=wp.vec4)
        opacities = wp.zeros(self.num_points, dtype=float)
        shs = wp.zeros(self.num_points * 16, dtype=wp.vec3)  # 16 coeffs per point
        camera_center = self.compute_initialization_center()
        # Launch kernel to initialize parameters
        wp.launch(
            init_gaussian_params,
            dim=self.num_points,
            inputs=[camera_center, positions, scales, rotations, opacities, shs, self.num_points, self.config['initial_scale']]
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
    
    def compute_initialization_center(self):
        """Compute central point from camera look-at positions."""
        centers = []
        for cam in self.cameras:
            forward = -cam['R'][:, 2]  # camera forward direction
            look_at = cam['camera_pos'] + forward * 1.0
            centers.append(look_at)
        
        scene_center = np.mean(centers, axis=0)
        return scene_center

    def load_nerf_data(self):
        """Load camera parameters and images from a NeRF dataset."""
        # Read transforms_train.json
        transforms_path = self.dataset_path / "transforms_train.json"
        if not transforms_path.exists():
            raise FileNotFoundError(f"No transforms_train.json found in {self.dataset_path}")
        
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        
        # Get image dimensions from the first image if available
        first_frame = transforms['frames'][0]
        first_img_path = str(self.dataset_path / f"{first_frame['file_path']}.png")
        if os.path.exists(first_img_path):
            # Load first image to get dimensions
            img = imageio.imread(first_img_path)
            width = img.shape[1]
            height = img.shape[0]
            print(f"Using image dimensions from dataset: {width}x{height}")
        else:
            # Use default dimensions from config if image not found
            width = self.config['width']
            height = self.config['height']
            print(f"Using default dimensions: {width}x{height}")
        
        # Update config with actual dimensions
        self.config['width'] = width
        self.config['height'] = height
        
        self.config['camera_angle_x'] = transforms['camera_angle_x']
        
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
            
            # Initialize camera dictionary
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
                'height': height,
                'depth_reliable': False,  # Initialize depth reliability flag
                'invdepthmap': None,      # Initialize inverse depth map
                'depth_mask': None        # Initialize depth mask
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
    
    def project_points(self, points_3d, view_matrix, proj_matrix, W, H):
        """
        Projects 3D points into image space (pixel xy and depth) using the given view and projection matrices.

        Args:
            points_3d (np.ndarray): (N, 3) array of 3D points in world coordinates
            view_matrix (np.ndarray): (4, 4) view matrix (world-to-camera)
            proj_matrix (np.ndarray): (4, 4) projection matrix (camera-to-clip)
            W (int): image width
            H (int): image height

        Returns:
            xy_image: (N, 2) projected pixel coordinates (in image space)
            depth: (N,) depth values in view space (i.e., z in camera space)
        """
        N = points_3d.shape[0]

        # Convert to homogeneous coordinates
        points_h = np.concatenate([points_3d, np.ones((N, 1))], axis=1)  # (N, 4)

        # Transform to view space
        view_points = (view_matrix @ points_h.T).T  # (N, 4)
        depth = view_points[:, 2]  # z in view space

        # Transform to clip space
        clip_points = (proj_matrix @ view_points.T).T  # (N, 4)
        clip_points /= clip_points[:, 3:4]  # perspective divide

        # Map to NDC [-1, 1]
        ndc = clip_points[:, :3]

        # Map NDC to image coordinates
        x_img = (ndc[:, 0] + 1) * 0.5 * W
        y_img = (ndc[:, 1] + 1) * 0.5 * H

        xy_image = np.stack([x_img, y_img], axis=1)  # (N, 2)

        return xy_image, depth

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
        rendered_image, _, _ = render_gaussians(
            background=np.array(self.config['background_color'], dtype=np.float32),
            means3D=self.params['positions'].numpy(),
            colors=None,  # Use SH coefficients instead
            opacity=self.params['opacities'].numpy(),
            scales=self.params['scales'].numpy(),
            rotations=self.params['rotations'].numpy(),
            scale_modifier=self.config['scale_modifier'],
            viewmatrix=self.cameras[camera_idx]['view_matrix'],
            projmatrix=self.cameras[camera_idx]['proj_matrix'],
            tan_fovx=self.cameras[camera_idx]['tan_fovx'],
            tan_fovy=self.cameras[camera_idx]['tan_fovy'],
            image_height=self.cameras[camera_idx]['height'],
            image_width=self.cameras[camera_idx]['width'],
            sh=self.params['shs'].numpy(),  # Pass SH coefficients
            degree=self.config['sh_degree'],
            campos=self.cameras[camera_idx]['camera_pos'],
            prefiltered=False,
            antialiasing=True,
            clamped=True
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(rendered_image)
        plt.title(f'Rendered View at Iteration {iteration}')
        plt.axis('off')
        plt.savefig(checkpoint_dir / "rendered_view.png")
        plt.close()
        
    def debug_log_and_save_images(self, rendered_image, target_image, depth_image, camera_idx, iteration):
        
        point_offsets = self.intermediate_buffers['point_offsets']
        num_rendered = int(wp.to_torch(point_offsets)[-1])
        print("duplicated entries (after radius / tile):", num_rendered)

        # 2) radii & opacity
        r_np  = wp.to_torch(self.intermediate_buffers['radii']).cpu().numpy()
        o_np  = wp.to_torch(self.intermediate_buffers['conic_opacity']).cpu().numpy()[:,3]   # 第 4 分量是 opacity
        print("radius  min/med/max:", r_np.min(), np.median(r_np[r_np>0]), r_np.max())
        print("opacity min/med/max:", o_np.min(), np.median(o_np), o_np.max())


        # save rendered image for debug
        # Convert to uint8 format before saving to avoid the data type error
        rendered_uint8 = (np.clip(rendered_image, 0, 1) * 255).astype(np.uint8)
        target_uint8 = (np.clip(target_image, 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(self.output_path / f"train_rendered_image_{iteration:06d}.png", rendered_uint8)
        imageio.imwrite(self.output_path / f"train_target_image_{iteration:06d}.png", target_uint8)
        
                
        image_width = self.cameras[camera_idx]['width']
        image_height = self.cameras[camera_idx]['height']
        
        xy = wp.to_torch(self.intermediate_buffers['points_xy_image']).cpu().numpy()
        plt.figure(figsize=(10, 10))
        plt.scatter(xy[:, 0], xy[:, 1], s=1)
        plt.xlim([0, image_width])
        plt.ylim([image_height, 0])
        plt.title("Projected 2D Gaussian Centers")
        plt.savefig(self.output_path / f"gaussian_centers_{iteration:06d}.png")
        plt.close()

        rgb = wp.to_torch(self.intermediate_buffers['rgb']).cpu().numpy()
        rgb = np.clip(rgb, 0.0, 1.0)
        plt.figure(figsize=(10, 10))
        plt.scatter(xy[:, 0], xy[:, 1], c=rgb, s=2)
        plt.title("Projected Gaussians Colored by SH Output")
        plt.savefig(self.output_path / f"gaussian_colors_{iteration:06d}.png")
        plt.close()
        
        xy = wp.to_torch(self.intermediate_buffers['points_xy_image']).cpu().numpy()
        rgb = wp.to_torch(self.intermediate_buffers['rgb']).cpu().numpy()
        rgb = np.clip(rgb, 0.0, 1.0)

        plt.figure(figsize=(10, 10))
        plt.scatter(xy[:, 0], xy[:, 1], c=rgb, s=2)
        plt.xlim([0, image_width])
        plt.ylim([image_height, 0])
        plt.title("Debug: Projected Gaussians with RGB")
        plt.savefig(self.output_path / f"debug_gaussians_rgb_{iteration:06d}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        points_xy = wp.to_torch(self.intermediate_buffers['points_xy_image']).cpu().numpy()
        image_w = self.config['width']
        image_h = self.config['height']
        mask_in_frame = (
            (points_xy[:, 0] >= 0) & (points_xy[:, 0] < image_w) &
            (points_xy[:, 1] >= 0) & (points_xy[:, 1] < image_h)
        )
        print(f"{np.sum(mask_in_frame)} / {len(points_xy)} Gaussians project inside the image bounds")

    
    def train(self):
        """Train the 3D Gaussian Splatting model."""
        num_iterations = self.config['num_iterations']
        
        # Main training loop
        with tqdm(total=num_iterations) as pbar:
            for iteration in range(num_iterations):
                # Select a random camera and corresponding image
                # camera_idx = np.random.randint(0, len(self.cameras))
                camera_idx = 3
                image_path = self.image_paths[camera_idx]
                target_image = self.load_image(image_path)
                
                # Zero gradients
                self.zero_grad()
                print("self.params['positions'].numpy()", self.params['positions'].numpy().shape)
                print("self.params['opacities'].numpy()", self.params['opacities'].numpy().shape)
                print("self.params['scales'].numpy()", self.params['scales'].numpy().shape)
                print("self.params['rotations'].numpy()", self.params['rotations'].numpy().shape)
                print("self.params['shs'].numpy()", self.params['shs'].numpy().shape)
                print("self.config['scale_modifier']", self.config['scale_modifier'])
                print("self.cameras[camera_idx]['view_matrix']", self.cameras[camera_idx]['view_matrix'])
                print("self.cameras[camera_idx]['proj_matrix']", self.cameras[camera_idx]['proj_matrix'])
                print("self.cameras[camera_idx]['tan_fovx']", self.cameras[camera_idx]['tan_fovx'])
                print("self.cameras[camera_idx]['tan_fovy']", self.cameras[camera_idx]['tan_fovy'])
                print("self.cameras[camera_idx]['height']", self.cameras[camera_idx]['height'])
                print("self.cameras[camera_idx]['width']", self.cameras[camera_idx]['width'])
                print("self.cameras[camera_idx]['camera_pos']", self.cameras[camera_idx]['camera_pos'])
                print("self.config['sh_degree']", self.config['sh_degree'])
                # Render the view
                rendered_image, depth_image, self.intermediate_buffers = render_gaussians(
                    background=np.array(self.config['background_color'], dtype=np.float32),
                    means3D=self.params['positions'].numpy(),
                    colors=None,  # Use SH coefficients instead
                    opacity=self.params['opacities'].numpy(),
                    scales=self.params['scales'].numpy(),
                    rotations=self.params['rotations'].numpy(),
                    scale_modifier=self.config['scale_modifier'],
                    viewmatrix=self.cameras[camera_idx]['view_matrix'],
                    projmatrix=self.cameras[camera_idx]['proj_matrix'],
                    tan_fovx=self.cameras[camera_idx]['tan_fovx'],
                    tan_fovy=self.cameras[camera_idx]['tan_fovy'],
                    image_height=self.cameras[camera_idx]['height'],
                    image_width=self.cameras[camera_idx]['width'],
                    sh=self.params['shs'].numpy(),  # Pass SH coefficients
                    degree=self.config['sh_degree'],
                    campos=self.cameras[camera_idx]['camera_pos'],
                    prefiltered=False,
                    antialiasing=False,
                    clamped=True
                )
                

                                
                
                self.debug_log_and_save_images(rendered_image, target_image, depth_image, camera_idx, iteration)
                
                exit()


  

                # Calculate L1 loss
                l1_val = l1_loss(rendered_image, target_image)
                
                # Calculate SSIM
                ssim_val = ssim(rendered_image, target_image)
                
                # Combined loss with weighted SSIM
                lambda_dssim = self.config['lambda_dssim']
                # loss = (1 - λ) * L1 + λ * (1 - SSIM)
                loss = (1.0 - lambda_dssim) * l1_val + lambda_dssim * (1.0 - ssim_val)
                self.losses.append(loss)
                
                # Compute pixel gradients for image loss (dL/dColor)
                pixel_grad_buffer = compute_image_gradients(
                    rendered_image, target_image, lambda_dssim=lambda_dssim
                )
                
                # Prepare camera parameters
                camera = self.cameras[camera_idx]
                view_matrix = wp.mat44(camera['view_matrix'].flatten())
                proj_matrix = wp.mat44(camera['proj_matrix'].flatten())
                campos = wp.vec3(camera['camera_pos'][0], camera['camera_pos'][1], camera['camera_pos'][2])

                # Create appropriate buffer dictionaries for the backward pass
                geom_buffer = {
                    'radii': self.intermediate_buffers['radii'],
                    'means2D': self.intermediate_buffers['points_xy_image'],
                    'conic_opacity': self.intermediate_buffers['conic_opacity'],
                    'rgb': self.intermediate_buffers['rgb'],
                    'clamped': self.intermediate_buffers['clamped_state']
                }
                
                binning_buffer = {
                    'point_list': self.intermediate_buffers['point_list']
                }
                
                img_buffer = {
                    'ranges': self.intermediate_buffers['ranges'],
                    'final_Ts': self.intermediate_buffers['final_Ts'],
                    'n_contrib': self.intermediate_buffers['n_contrib']
                }
                
                gradients = backward(
                    # Core parameters
                    background=np.array(self.config['background_color'], dtype=np.float32),
                    means3D=self.params['positions'],
                    dL_dpixels=pixel_grad_buffer,
                    # dL_invdepths=depth_grad_buffer,  # Pass depth gradients
                    # use_invdepth=True,
                    
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
                    radii=self.intermediate_buffers['radii'],
                    means2D=self.intermediate_buffers['points_xy_image'],
                    conic_opacity=self.intermediate_buffers['conic_opacity'],
                    rgb=self.intermediate_buffers['rgb'],
                    depth=self.intermediate_buffers['depths'],
                    cov3Ds=self.intermediate_buffers['cov3Ds'],
                    clamped=self.intermediate_buffers['clamped_state'],
                    
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

                # Convert Warp arrays to numpy for statistics
                pos_np = self.grads['positions'].numpy()
                scales_np = self.grads['scales'].numpy()
                rot_np = self.grads['rotations'].numpy()
                opac_np = self.grads['opacities'].numpy()
                shs_np = self.grads['shs'].numpy()
                
                print("self.grads['positions']", self.grads['positions'], np.max(pos_np), np.min(pos_np))
                print("self.grads['scales']", self.grads['scales'], np.max(scales_np), np.min(scales_np))
                print("self.grads['rotations']", self.grads['rotations'], np.max(rot_np), np.min(rot_np))
                print("self.grads['opacities']", self.grads['opacities'], np.max(opac_np), np.min(opac_np))
                print("self.grads['shs']", self.grads['shs'], np.max(shs_np), np.min(shs_np))
                print("Rendered image mean:", wp.to_torch(rendered_image).mean().item(), wp.to_torch(rendered_image).max().item(), wp.to_torch(rendered_image).min().item())
                print("Pixel gradient mean:", wp.to_torch(pixel_grad_buffer).abs().mean(), wp.to_torch(pixel_grad_buffer).abs().max(), wp.to_torch(pixel_grad_buffer).abs().min())
                
                exit()
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

    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = NeRFGaussianSplattingTrainer(
        dataset_path=args.dataset,
        output_path=args.output,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
