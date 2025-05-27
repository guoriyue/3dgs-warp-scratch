import os
import numpy as np
import matplotlib.pyplot as plt
import warp as wp
import imageio
import json
from tqdm import tqdm
from pathlib import Path
import argparse

# Import the renderer and constants
from forward import render_gaussians
from backward import (backward, prune_gaussians, adam_update, clone_gaussians, compact_gaussians,
                     mark_split_candidates, mark_clone_candidates,
                     split_gaussians, reset_opacities, reset_densification_stats)
from config import *
from utils.camera_utils import load_camera
from utils.point_cloud_utils import save_ply
from loss import l1_loss, compute_image_gradients

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
    # Generate random positions using warp random
    offset = wp.vec3(
        (wp.randf(wp.uint32(i * 3)) * 2.6 - 1.3),
        (wp.randf(wp.uint32(i * 3 + 1)) * 2.6 - 1.3),
        (wp.randf(wp.uint32(i * 3 + 2)) * 2.6 - 1.3)
    )
    # camera_center
    positions[i] =  offset
    
    # Initialize scales
    scales[i] = wp.vec3(init_scale, init_scale, init_scale)
    
    # Initialize rotations to identity matrix
    rotations[i] = wp.vec4(1.0, 0.0, 0.0, 0.0)
    
    # Initialize opacities
    opacities[i] = 0.1
    
    # Initialize SH coefficients (just DC term for now)
    for j in range(16):  # degree=3, total 16 coefficients
        idx = i * 16 + j
        # Slight random initialization with positive bias
        if j == 0:
            shs[idx] = wp.vec3(-0.007, -0.007, -0.007)
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
        self.cameras, self.image_paths = self.load_nerf_data("train")
        self.val_cameras, self.val_image_paths = self.load_nerf_data("val")
        self.test_cameras, self.test_image_paths = self.load_nerf_data("test")
        print(f"Loaded {len(self.cameras)} train cameras and {len(self.image_paths)} train images")
        print(f"Loaded {len(self.val_cameras)} val cameras and {len(self.val_image_paths)} val images")
        print(f"Loaded {len(self.test_cameras)} test cameras and {len(self.test_image_paths)} test images")
        
        # Calculate scene extent for densification
        self.scene_extent = self.calculate_scene_extent()
        print(f"Calculated scene extent: {self.scene_extent}")
        
        # Initialize parameters
        self.num_points = self.config['num_points']
        self.params = self.initialize_parameters()
        
        # Create gradient arrays
        self.grads = self.create_gradient_arrays()
        
        # Create optimizer state
        self.adam_m = self.create_gradient_arrays()  # First moment
        self.adam_v = self.create_gradient_arrays()  # Second moment
        
        # Initialize densification state tracking
        self.init_densification_state()
        
        # For tracking loss
        self.losses = []
        
        # Initialize intermediate buffers dictionary
        self.intermediate_buffers = {}
        
        # Track iteration for opacity reset
        self.opacity_reset_at = -32768
    
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

    def calculate_scene_extent(self):
        """Calculate the extent of the scene based on camera positions."""
        if not self.cameras:
            return 1.0  # Default fallback
        
        # Extract camera positions
        camera_positions = []
        for camera in self.cameras:
            camera_positions.append(camera['camera_center'])
        
        camera_positions = np.array(camera_positions)
        
        # Calculate the extent as the maximum distance between any two cameras
        max_extent = 0.0
        for i in range(len(camera_positions)):
            for j in range(i + 1, len(camera_positions)):
                dist = np.linalg.norm(camera_positions[i] - camera_positions[j])
                max_extent = max(max_extent, dist)
        
        # Use default factor if extent is too small
        return max(max_extent * self.config.get('camera_extent_factor', 1.0), 1.0)

    def init_densification_state(self):
        """Initialize state tracking for densification."""
        self.xyz_gradient_accum = wp.zeros(self.num_points, dtype=float, device=DEVICE)
        self.denom = wp.zeros(self.num_points, dtype=float, device=DEVICE)
        self.max_radii2D = wp.zeros(self.num_points, dtype=float, device=DEVICE)

    def reset_densification_state(self):
        """Reset densification state after parameter count changes."""
        wp.launch(
            reset_densification_stats,
            dim=self.num_points,
            inputs=[
                self.xyz_gradient_accum,
                self.denom,
                self.max_radii2D,
                self.num_points
            ]
        )

    def load_nerf_data(self, datasplit):
        """Load camera parameters and images from a NeRF dataset."""
        # Read transforms_train.json
        transforms_path = self.dataset_path / f"transforms_{datasplit}.json"
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
            camera_info = {
                "camera_id": i,
                "camera_to_world": frame['transform_matrix'],
                "width": width,
                "height": height,
                "focal": focal,
            }
            
            # Load camera parameters using existing function
            camera_params = load_camera(camera_info)
            
            
            if camera_params is not None:
                cameras.append(camera_params)
                image_paths.append(str(self.dataset_path / f"{frame['file_path']}.png"))
        
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
        """Perform sophisticated densification and pruning of Gaussians."""
        
        # Check if we should do densification
        densify_from_iter = self.config.get('densify_from_iter', 500)
        densify_until_iter = self.config.get('densify_until_iter', 15000)
        densification_interval = self.config.get('densification_interval', 100)
        opacity_reset_interval = self.config.get('opacity_reset_interval', 3000)
        
        # Opacity reset
        if (iteration % opacity_reset_interval == 0 or
            iteration == densify_from_iter):
            print(f"Iteration {iteration}: Resetting opacities")
            wp.launch(
                reset_opacities,
                dim=self.num_points,
                inputs=[
                    self.params['opacities'],
                    0.01,  # max_opacity
                    self.num_points
                ]
            )
        
        # Skip densification if outside iteration range
        if iteration < densify_from_iter or iteration >= densify_until_iter:
            return
        
        # Only densify at specified intervals
        if iteration % densification_interval != 0:
            return
            
        print(f"Iteration {iteration}: Performing sophisticated densification and pruning")
        
        # For simplified implementation, use position gradients as proxy for viewspace gradients
        pos_grads = self.grads['positions']
        avg_grads = wp.zeros(self.num_points, dtype=float, device=DEVICE)
        
        @wp.kernel
        def compute_grad_norms(pos_grad: wp.array(dtype=wp.vec3),
                              grad_norms: wp.array(dtype=float),
                              num_points: int):
            i = wp.tid()
            if i >= num_points:
                return
            grad_norms[i] = wp.length(pos_grad[i])
        
        wp.launch(compute_grad_norms, dim=self.num_points,
                 inputs=[pos_grads, avg_grads, self.num_points])
        
        # Configuration
        grad_threshold = self.config.get('densify_grad_threshold', 0.0002)
        percent_dense = self.config.get('percent_dense', 0.01)
        
        # --- Step 1: Clone small Gaussians with high gradients ---
        clone_mask = wp.zeros(self.num_points, dtype=int, device=DEVICE)
        wp.launch(
            mark_clone_candidates,
            dim=self.num_points,
            inputs=[
                avg_grads,
                self.params['scales'],
                grad_threshold,
                self.scene_extent,
                percent_dense,
                clone_mask,
                self.num_points
            ]
        )
        
        # Perform cloning
        clone_prefix_sum = wp.zeros_like(clone_mask)
        wp.utils.array_scan(clone_mask, clone_prefix_sum, inclusive=False)
        total_to_clone = int(clone_prefix_sum.numpy()[-1])
        
        if total_to_clone > 0:
            print(f"[Clone] Cloning {total_to_clone} small Gaussians")
            N = self.num_points
            new_N = N + total_to_clone
            
            # Allocate output arrays
            out_params = {
                'positions': wp.zeros(new_N, dtype=wp.vec3, device=DEVICE),
                'scales': wp.zeros(new_N, dtype=wp.vec3, device=DEVICE),
                'rotations': wp.zeros(new_N, dtype=wp.vec4, device=DEVICE),
                'opacities': wp.zeros(new_N, dtype=float, device=DEVICE),
                'shs': wp.zeros(new_N * 16, dtype=wp.vec3, device=DEVICE)
            }
            
            # Clone Gaussians
            wp.launch(
                clone_gaussians,
                dim=N,
                inputs=[
                    clone_mask,
                    clone_prefix_sum,
                    self.params['positions'],
                    self.params['scales'],
                    self.params['rotations'],
                    self.params['opacities'],
                    self.params['shs'],
                    0.01,  # noise_scale
                    N,     # offset
                    out_params['positions'],
                    out_params['scales'],
                    out_params['rotations'],
                    out_params['opacities'],
                    out_params['shs']
                ]
            )
            
            # Update parameters and state
            self.params = out_params
            self.num_points = new_N
            self.grads = self.create_gradient_arrays()
            self.adam_m = self.create_gradient_arrays()
            self.adam_v = self.create_gradient_arrays()
        
        # --- Step 2: Split large Gaussians with high gradients ---
        split_mask = wp.zeros(self.num_points, dtype=int, device=DEVICE)
        wp.launch(
            mark_split_candidates,
            dim=self.num_points,
            inputs=[
                avg_grads,
                self.params['scales'],
                grad_threshold,
                self.scene_extent,
                percent_dense,
                split_mask,
                self.num_points
            ]
        )
        
        # Perform splitting
        split_prefix_sum = wp.zeros_like(split_mask)
        wp.utils.array_scan(split_mask, split_prefix_sum, inclusive=False)
        total_to_split = int(split_prefix_sum.numpy()[-1])
        
        if total_to_split > 0:
            print(f"[Split] Splitting {total_to_split} large Gaussians")
            N = self.num_points
            N_split = 2  # Split each Gaussian into 2
            new_N = N + total_to_split * N_split
            
            # Allocate output arrays
            out_params = {
                'positions': wp.zeros(new_N, dtype=wp.vec3, device=DEVICE),
                'scales': wp.zeros(new_N, dtype=wp.vec3, device=DEVICE),
                'rotations': wp.zeros(new_N, dtype=wp.vec4, device=DEVICE),
                'opacities': wp.zeros(new_N, dtype=float, device=DEVICE),
                'shs': wp.zeros(new_N * 16, dtype=wp.vec3, device=DEVICE)
            }
            
            # Split Gaussians
            wp.launch(
                split_gaussians,
                dim=N,
                inputs=[
                    split_mask,
                    split_prefix_sum,
                    self.params['positions'],
                    self.params['scales'],
                    self.params['rotations'],
                    self.params['opacities'],
                    self.params['shs'],
                    N_split,  # Number of splits per Gaussian
                    0.8,      # scale_factor
                    N,        # offset
                    out_params['positions'],
                    out_params['scales'],
                    out_params['rotations'],
                    out_params['opacities'],
                    out_params['shs']
                ]
            )
            
            # Update parameters and state
            self.params = out_params
            self.num_points = new_N
            self.grads = self.create_gradient_arrays()
            self.adam_m = self.create_gradient_arrays()
            self.adam_v = self.create_gradient_arrays()
            
            # Remove original split Gaussians
            prune_filter = wp.zeros(self.num_points, dtype=int, device=DEVICE)
            
            @wp.kernel
            def mark_split_originals_for_removal(
                split_mask: wp.array(dtype=int),
                prune_filter: wp.array(dtype=int),
                offset: int,
                num_points: int
            ):
                i = wp.tid()
                if i >= num_points:
                    return
                if i < offset and split_mask[i] == 1:
                    prune_filter[i] = 1  # Mark for removal
                else:
                    prune_filter[i] = 0  # Keep
            
            wp.launch(mark_split_originals_for_removal, dim=self.num_points,
                     inputs=[split_mask, prune_filter, N, self.num_points])
            
            # Invert mask to get valid mask
            valid_mask = wp.zeros_like(prune_filter)
            
            @wp.kernel
            def invert_mask(prune: wp.array(dtype=int), valid: wp.array(dtype=int), n: int):
                i = wp.tid()
                if i >= n:
                    return
                valid[i] = 1 - prune[i]
            
            wp.launch(invert_mask, dim=self.num_points, 
                     inputs=[prune_filter, valid_mask, self.num_points])
            
            # Count valid points and compact
            prefix_sum = wp.zeros_like(valid_mask)
            wp.utils.array_scan(valid_mask, prefix_sum, inclusive=False)
            valid_count = int(prefix_sum.numpy()[-1])
            
            if valid_count < self.num_points:
                print(f"[Split] Removing {self.num_points - valid_count} original split Gaussians")
                
                # Allocate compacted output
                compact_params = {
                    'positions': wp.zeros(valid_count, dtype=wp.vec3, device=DEVICE),
                    'scales': wp.zeros(valid_count, dtype=wp.vec3, device=DEVICE),
                    'rotations': wp.zeros(valid_count, dtype=wp.vec4, device=DEVICE),
                    'opacities': wp.zeros(valid_count, dtype=float, device=DEVICE),
                    'shs': wp.zeros(valid_count * 16, dtype=wp.vec3, device=DEVICE)
                }
                
                wp.launch(
                    compact_gaussians,
                    dim=self.num_points,
                    inputs=[
                        valid_mask,
                        prefix_sum,
                        self.params['positions'],
                        self.params['scales'],
                        self.params['rotations'],
                        self.params['opacities'],
                        self.params['shs'],
                        compact_params['positions'],
                        compact_params['scales'],
                        compact_params['rotations'],
                        compact_params['opacities'],
                        compact_params['shs']
                    ]
                )
                
                # Update parameters and state
                self.params = compact_params
                self.num_points = valid_count
                self.grads = self.create_gradient_arrays()
                self.adam_m = self.create_gradient_arrays()
                self.adam_v = self.create_gradient_arrays()
        
        # --- Step 3: Enhanced Pruning ---
        print(f"[Prune] Performing enhanced pruning")
        
        valid_mask = wp.zeros(self.num_points, dtype=int, device=DEVICE)
        
        # Use opacity-based pruning for now
        wp.launch(
            prune_gaussians,
            dim=self.num_points,
            inputs=[
                self.params['opacities'],
                self.config.get('cull_opacity_threshold', 0.005),
                valid_mask,
                self.num_points
            ]
        )
        
        # Count valid points
        prefix_sum = wp.zeros_like(valid_mask)
        wp.utils.array_scan(valid_mask, prefix_sum, inclusive=False)
        valid_count = int(prefix_sum.numpy()[-1])
        
        # Check pruning constraints
        min_valid_points = self.config.get('min_valid_points', 1000)
        max_valid_points = self.config.get('max_valid_points', 1000000)
        max_prune_ratio = self.config.get('max_allowed_prune_ratio', 0.5)
        
        prune_count = self.num_points - valid_count
        prune_ratio = prune_count / self.num_points if self.num_points > 0 else 0
        
        if (valid_count >= min_valid_points and 
            valid_count <= max_valid_points and 
            prune_ratio <= max_prune_ratio and
            valid_count < self.num_points):
            
            print(f"[Prune] Compacting from {self.num_points} → {valid_count} points")
            
            # Allocate compacted output
            out_params = {
                'positions': wp.zeros(valid_count, dtype=wp.vec3, device=DEVICE),
                'scales': wp.zeros(valid_count, dtype=wp.vec3, device=DEVICE),
                'rotations': wp.zeros(valid_count, dtype=wp.vec4, device=DEVICE),
                'opacities': wp.zeros(valid_count, dtype=float, device=DEVICE),
                'shs': wp.zeros(valid_count * 16, dtype=wp.vec3, device=DEVICE)
            }
            
            wp.launch(
                compact_gaussians,
                dim=self.num_points,
                inputs=[
                    valid_mask,
                    prefix_sum,
                    self.params['positions'],
                    self.params['scales'],
                    self.params['rotations'],
                    self.params['opacities'],
                    self.params['shs'],
                    out_params['positions'],
                    out_params['scales'],
                    out_params['rotations'],
                    out_params['opacities'],
                    out_params['shs']
                ]
            )
            
            # Update parameters and state
            self.params = out_params
            self.num_points = valid_count
            self.grads = self.create_gradient_arrays()
            self.adam_m = self.create_gradient_arrays()
            self.adam_v = self.create_gradient_arrays()
        else:
            print(f"[Prune] Skipping pruning: valid={valid_count}, ratio={prune_ratio:.3f}")

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
                self.config['lr_pos'],
                self.config['lr_scale'],
                self.config['lr_rot'],
                self.config['lr_sh'],
                self.config['lr_opac'],
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
            viewmatrix=self.cameras[camera_idx]['world_to_camera'],
            projmatrix=self.cameras[camera_idx]['full_proj_matrix'],
            tan_fovx=self.cameras[camera_idx]['tan_fovx'],
            tan_fovy=self.cameras[camera_idx]['tan_fovy'],
            image_height=self.cameras[camera_idx]['height'],
            image_width=self.cameras[camera_idx]['width'],
            sh=self.params['shs'].numpy(),  # Pass SH coefficients
            degree=self.config['sh_degree'],
            campos=self.cameras[camera_idx]['camera_center'],
            prefiltered=False,
            antialiasing=True,
            clamped=True
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(wp.to_torch(rendered_image).cpu().numpy())
        plt.title(f'Rendered View at Iteration {iteration}')
        plt.axis('off')
        plt.savefig(checkpoint_dir / "rendered_view.png")
        plt.close()
        

    def debug_log_and_save_images(
            self,
            rendered_image,         # np.float32  H×W×3  (range 0-1)
            target_image,           # np.float32
            depth_image,            # wp.array2d(float) – optional but unused here
            camera_idx: int,
            it: int
    ):

        # ------ quick numeric read-out -----------------------------------
        radii   = wp.to_torch(self.intermediate_buffers["radii"]).cpu().numpy()
        alphas  = wp.to_torch(self.intermediate_buffers["conic_opacity"]).cpu().numpy()[:, 3]
        offs    = wp.to_torch(self.intermediate_buffers["point_offsets"]).cpu().numpy()
        num_dup = int(offs[-1]) if len(offs) else 0
        r_med   = np.median(radii[radii > 0]) if (radii > 0).any() else 0
        
        # Count visible Gaussians
        xy_image = wp.to_torch(self.intermediate_buffers["points_xy_image"]).cpu().numpy()
        W = self.cameras[camera_idx]['width']
        H = self.cameras[camera_idx]['height']
        visible_gaussians = np.sum(
            (xy_image[:, 0] >= 0) & (xy_image[:, 0] < W) & 
            (xy_image[:, 1] >= 0) & (xy_image[:, 1] < H) &
            np.isfinite(xy_image).all(axis=1) &
            (radii > 0)  # Only count Gaussians with positive radius
        )
        
        print(
            f"[it {it:05d}] dup={num_dup:<6} "
            f"r_med={r_med:5.1f}  α∈[{alphas.min():.3f},"
            f"{np.median(alphas):.3f},{alphas.max():.3f}] "
            f"visible={visible_gaussians}/{len(xy_image)}"
        )

        # ------ save render / target PNG ---------------------------------
        def save_rgb(arr_f32, stem):
            # Handle case where arr_f32 has shape (3, H, W) - transpose to (H, W, 3)
            if arr_f32.shape[0] == 3 and len(arr_f32.shape) == 3:
                arr_f32 = np.transpose(arr_f32, (1, 2, 0))
            img8 = (np.clip(arr_f32, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(self.output_path / f"{stem}_{it:06d}.png", img8)
        
        save_rgb(rendered_image if isinstance(rendered_image, np.ndarray) else wp.to_torch(rendered_image).cpu().numpy(), "render")
        save_rgb(target_image,   "target")

        # ------ make 2-D projection scatter ------------------------------
        xy     = wp.to_torch(self.intermediate_buffers["points_xy_image"]).cpu().numpy()
        depth  = wp.to_torch(self.intermediate_buffers["depths"]).cpu().numpy()
        H, W   = self.config["height"], self.config["width"]

        mask = (
            (xy[:, 0] >= 0) & (xy[:, 0] < W) &
            (xy[:, 1] >= 0) & (xy[:, 1] < H) &
            np.isfinite(xy).all(axis=1) &
            (radii > 0)  # Only include Gaussians with positive radius
        )
        if mask.any():
            plt.figure(figsize=(6, 6))
            plt.scatter(xy[mask, 0], xy[mask, 1],
                        s=4, c=depth[mask], cmap="turbo", alpha=.7)
            plt.gca().invert_yaxis()
            plt.xlim(0, W); plt.ylim(H, 0)
            plt.title(f"Projected Gaussians (iter {it}): {np.sum(mask)}/{len(xy)} visible")
            plt.colorbar(label="depth(z)")
            plt.tight_layout()
            plt.savefig(self.output_path / f"proj_{it:06d}.png", dpi=250)
            plt.close()

            # depth histogram
            plt.figure(figsize=(5, 3))
            plt.hist(depth[mask], bins=40, color="steelblue")
            plt.xlabel("depth (camera-z)")
            plt.ylabel("count")
            plt.title(f"Depth hist – {mask.sum()} pts")
            plt.tight_layout()
            plt.savefig(self.output_path / f"depth_hist_{it:06d}.png", dpi=250)
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
                rendered_image, depth_image, self.intermediate_buffers = render_gaussians(
                    background=np.array(self.config['background_color'], dtype=np.float32),
                    means3D=self.params['positions'].numpy(),
                    colors=None,  # Use SH coefficients instead
                    opacity=self.params['opacities'].numpy(),
                    scales=self.params['scales'].numpy(),
                    rotations=self.params['rotations'].numpy(),
                    scale_modifier=self.config['scale_modifier'],
                    viewmatrix=self.cameras[camera_idx]['world_to_camera'],
                    projmatrix=self.cameras[camera_idx]['full_proj_matrix'],
                    tan_fovx=self.cameras[camera_idx]['tan_fovx'],
                    tan_fovy=self.cameras[camera_idx]['tan_fovy'],
                    image_height=self.cameras[camera_idx]['height'],
                    image_width=self.cameras[camera_idx]['width'],
                    sh=self.params['shs'].numpy(),  # Pass SH coefficients
                    degree=self.config['sh_degree'],
                    campos=self.cameras[camera_idx]['camera_center'],
                    prefiltered=False,
                    antialiasing=False,
                    clamped=True
                )

                radii = wp.to_torch(self.intermediate_buffers["radii"]).cpu().numpy()
                np_rendered_image = wp.to_torch(rendered_image).cpu().numpy()
                np_rendered_image = np_rendered_image.transpose(2, 0, 1)

                if iteration % self.config['save_interval'] == 0:
                    self.debug_log_and_save_images(np_rendered_image, target_image, depth_image, camera_idx, iteration)

                # Calculate L1 loss
                l1_val = l1_loss(rendered_image, target_image)
                
                # # Calculate SSIM, not used
                # ssim_val = ssim(rendered_image, target_image)
                # # Combined loss with weighted SSIM
                # lambda_dssim = self.config['lambda_dssim']
                # # loss = (1 - λ) * L1 + λ * (1 - SSIM)
                # loss = (1.0 - lambda_dssim) * l1_val + lambda_dssim * (1.0 - ssim_val)
                
                loss = l1_val
                self.losses.append(loss)
                # Compute pixel gradients for image loss (dL/dColor)
                pixel_grad_buffer = compute_image_gradients(
                    rendered_image, target_image, lambda_dssim=0
                )
                
                # Prepare camera parameters
                camera = self.cameras[camera_idx]
                view_matrix = wp.mat44(camera['world_to_camera'].flatten())
                proj_matrix = wp.mat44(camera['full_proj_matrix'].flatten())
                campos = wp.vec3(camera['camera_center'][0], camera['camera_center'][1], camera['camera_center'][2])

                # Create appropriate buffer dictionaries for the backward pass
                geom_buffer = {
                    'radii': self.intermediate_buffers['radii'],
                    'means2D': self.intermediate_buffers['points_xy_image'],
                    'conic_opacity': self.intermediate_buffers['conic_opacity'],
                    'rgb': self.intermediate_buffers['colors'],
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
                    rgb=self.intermediate_buffers['colors'],
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

                # Update parameters
                self.optimizer_step(iteration)
     
                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Loss: {loss:.6f}")
                
                self.densification_and_pruning(iteration)
                
                # Save checkpoint
                if iteration % self.config['save_interval'] == 0 or iteration == num_iterations - 1:
                    self.save_checkpoint(iteration)
                
        print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting model with NeRF dataset")
    parser.add_argument("--dataset", type=str, default="./data/nerf_synthetic/lego",
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
