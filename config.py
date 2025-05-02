"""
Configuration settings and constants for 3D Gaussian Splatting with NeRF datasets.
"""
import warp as wp
import torch
import numpy as np


# Warp data types and constants (keep capitalized as they are types)
WP_FLOAT16 = wp.float16
WP_FLOAT32 = wp.float32
WP_INT = wp.int32
WP_VEC2 = wp.vec2
WP_VEC2H = wp.vec2h
VEC6 = wp.types.vector(length=6, dtype=WP_FLOAT32)
TORCH_FLOAT = torch.float32
DEVICE = "cpu" # Use "cpu" or "cuda"

TILE_M = wp.constant(128)
TILE_N = wp.constant(128)
TILE_THREADS = wp.constant(256)

class RenderParams:
    """Parameters for rendering."""
    
    # Default image dimensions
    default_width = 1800
    default_height = 1800
    
    # Camera parameters
    default_fovx = 45.0  # degrees
    default_fovy = 45.0  # degrees
    default_znear = 0.01
    default_zfar = 100.0
    
    # Example camera position
    default_camera_pos = np.array([0, 0, 5], dtype=np.float32)
    default_camera_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    
    # Rendering parameters
    background_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Black background
    scale_modifier = 1.0
    prefiltered = False
    antialiasing = False
    clamped = True
    debug = False
    
    # SH degree for rendering
    sh_degree = 3

class GaussianParams:
    """Parameters for 3D Gaussian Splatting."""

    # Training parameters
    num_iterations = 1000  # Default number of training iterations
    learning_rate = 0.01  # Learning rate for Adam optimizer
    num_points = 5000  # Initial number of Gaussian points

    # Optimization parameters
    densification_interval = 300  # Perform densification every N iterations
    pruning_interval = 300  # Perform pruning every N iterations
    save_interval = 300  # Save checkpoint every N iterations
    adam_beta1 = 0.9  # Adam optimizer beta1 parameter
    adam_beta2 = 0.999  # Adam optimizer beta2 parameter
    adam_epsilon = 1e-8  # Adam optimizer epsilon parameter

    # Gaussian parameters
    initial_scale = 0.1  # Initial scale for Gaussian points
    scale_modifier = 1.0  # Scaling factor for Gaussian splats
    sh_degree = 3  # Spherical harmonics degree

    # Scene parameters
    scene_scale = 1.0  # Scale factor for the scene
    background_color = [1.0, 1.0, 1.0]  # White background for NeRF synthetic

    # Loss parameters
    # lambda_dssim = 0.2  # Weight for SSIM loss (1.0 means only SSIM, 0.0 means only L1)
    lambda_dssim = 0.0
    
    near = 0.01  # Default near clipping plane
    far = 100.0  # Default far clipping plane

    @classmethod
    def update(cls, **kwargs):
        """Update parameters with new values."""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    @classmethod
    def get_config_dict(cls):
        """Get parameters as a dictionary."""
        return {
            'num_iterations': cls.num_iterations,
            'learning_rate': cls.learning_rate,
            'num_points': cls.num_points,
            'densification_interval': cls.densification_interval,
            'pruning_interval': cls.pruning_interval,
            'scale_modifier': cls.scale_modifier,
            'sh_degree': cls.sh_degree,
            'background_color': cls.background_color,
            'save_interval': cls.save_interval,
            'adam_beta1': cls.adam_beta1,
            'adam_beta2': cls.adam_beta2,
            'adam_epsilon': cls.adam_epsilon,
            'initial_scale': cls.initial_scale,
            'scene_scale': cls.scene_scale,
            'near': cls.near,
            'far': cls.far,
            'lambda_dssim': cls.lambda_dssim
        }

