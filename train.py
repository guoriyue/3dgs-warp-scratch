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
        camera_center, avg_back_dir = self.compute_initialization_center()
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
        """
        A better heuristic for the scene centre:

        • For every camera, take its position P and its forward direction F
        (negative Z axis in NeRF-synthetic).
        • March 1.0 world-unit along F (so P + F) – that’s roughly the object surface.
        • Average those points.  Result ≈ centre of the lego excavator.
        • Return both the average centre and the average *backward* direction
        (useful if you later need a canonical ‘view’ dir).
        """
        pts_on_object = []
        back_dirs     = []

        for cam in self.cameras:
            P = np.asarray(cam["camera_pos"], dtype=np.float32)

            # camera forward is -R[:,2]  (NeRF convention)
            fwd = -cam["R"][:, 2]
            fwd /= np.linalg.norm(fwd)

            pts_on_object.append(P + fwd * 1.0)   # 1-unit in front
            back_dirs.append(-fwd)                # 'scene → camera' dir for later

        scene_center = np.mean(pts_on_object, axis=0)
        avg_back_dir = np.mean(back_dirs, axis=0)
        avg_back_dir /= np.linalg.norm(avg_back_dir)

        print("Better scene centre :", scene_center)
        return scene_center, avg_back_dir


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
            
            # FIX: Adjust view matrix to match the forward renderer expectation
            # Inverting the sign of the 3rd row to flip Z direction
            # This makes OpenGL-style coordinate systems work correctly with our renderer
            view_matrix = world2cam.copy()
            view_matrix[2, :] = -view_matrix[2, :]  # Flip Z row
            
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
        print(
            f"[it {it:05d}] dup={num_dup:<6} "
            f"r_med={r_med:5.1f}  α∈[{alphas.min():.3f},"
            f"{np.median(alphas):.3f},{alphas.max():.3f}]"
        )

        # ------ save render / target PNG ---------------------------------
        def save_rgb(arr_f32, stem):
            img8 = (np.clip(arr_f32, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(self.output_path / f"{stem}_{it:06d}.png", img8)

        save_rgb(rendered_image, "render")
        save_rgb(target_image,   "target")

        # ------ make 2-D projection scatter ------------------------------
        xy     = wp.to_torch(self.intermediate_buffers["points_xy_image"]).cpu().numpy()
        depth  = wp.to_torch(self.intermediate_buffers["depths"]).cpu().numpy()
        H, W   = self.config["height"], self.config["width"]

        mask = (
            (xy[:, 0] >= 0) & (xy[:, 0] < W) &
            (xy[:, 1] >= 0) & (xy[:, 1] < H) &
            np.isfinite(xy).all(axis=1)
        )
        if mask.any():
            plt.figure(figsize=(6, 6))
            plt.scatter(xy[mask, 0], xy[mask, 1],
                        s=4, c=depth[mask], cmap="turbo", alpha=.7)
            plt.gca().invert_yaxis()
            plt.xlim(0, W); plt.ylim(H, 0)
            plt.title(f"Projected Gaussians (iter {it})")
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
