import numpy as np
import warp as wp
from config import *
# Initialize Warp
wp.init()

syncthreads_snippet = """
__syncthreads();
"""

@wp.func_native(syncthreads_snippet)
def syncthreads():
    return

@wp.struct
class Gaussian3D:
    """Structure representing a 3D Gaussian."""
    mean: wp.vec3
    cov: wp.mat33
    color: wp.vec3
    opacity: float
    
@wp.struct
class Gaussian2D:
    """Structure representing a projected 2D Gaussian."""
    mean: wp.vec2
    cov: wp.mat22
    color: wp.vec3
    opacity: float
    depth: float

@wp.func
def compute_cov2d(cov3d: wp.mat33, view_matrix: wp.mat44, proj_matrix: wp.mat44, 
                 tan_fovx: float, tan_fovy: float, width: float, height: float) -> wp.mat22:
    # Extract rotation from view matrix (assuming no scaling)
    rot = wp.mat33(
        view_matrix[0, 0], view_matrix[0, 1], view_matrix[0, 2],
        view_matrix[1, 0], view_matrix[1, 1], view_matrix[1, 2],
        view_matrix[2, 0], view_matrix[2, 1], view_matrix[2, 2]
    )
    
    # Transform covariance to camera space
    cov_camera = wp.mul(wp.mul(rot, cov3d), wp.transpose(rot))
    
    # Compute Jacobian of projection
    fx = width / (2.0 * tan_fovx)
    fy = height / (2.0 * tan_fovy)
    
    # Create a simple projection Jacobian
    J = mat23(
        fx, 0.0, 0.0,
        0.0, fy, 0.0
    )
    
    # Project the covariance to 2D
    cov_image = wp.mat22()
    JW = mat23()
    JW = J * cov_camera
    cov_image = JW * J.T
    return cov_image



@wp.kernel
def wp_rasterize_gaussians(
    # Output buffers
    rendered_image: wp.array2d(dtype=wp.vec3),
    depth_image: wp.array2d(dtype=float),
    
    # Input parameters
    background: wp.vec3,
    means3d: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
    opacities: wp.array(dtype=float),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.vec4),  # Quaternions
    scale_modifier: float,
    
    # View parameters
    view_matrix: wp.mat44,
    proj_matrix: wp.mat44,
    tan_fovx: float,
    tan_fovy: float,
    
    # Camera parameters
    camera_pos: wp.vec3,
    
    # Additional flags
    antialiasing: int
):
    # Get thread indices
    pixel_x, pixel_y = wp.tid()
    
    # Initialize pixel color and depth
    color = background
    depth = float(1.0)
    accumulated_alpha = float(0.0)
    
    # Get image dimensions
    image_height = rendered_image.shape[0]
    image_width = rendered_image.shape[1]
    
    # Current pixel position
    pixel_pos = wp.vec2(float(pixel_x) + 0.5, float(pixel_y) + 0.5)
    
    # For each Gaussian
    for i in range(means3d.shape[0]):
        # Get Gaussian parameters
        mean3d = means3d[i]
        gaussian_color = colors[i]
        opacity = opacities[i]
        scale = scales[i] * scale_modifier
        rotations_i_quat = wp.types.quaternion(dtype=WP_FLOAT32)
        rotations_i_quat.x = rotations[i][0]
        rotations_i_quat.y = rotations[i][1]
        rotations_i_quat.z = rotations[i][2]
        rotations_i_quat.w = rotations[i][3]
        quat = wp.quaternion(rotations_i_quat, dtype=WP_FLOAT32)
        
        # Project 3D Gaussian to 2D
        # 1. Transform to camera space
        p_cam = wp.transform_point(view_matrix, mean3d)
        
        # Skip if behind camera
        if p_cam[2] <= 0.0:
            continue
            
        # 2. Project to clip space
        p_clip = wp.transform_point(proj_matrix, p_cam)
        
        # 3. Perspective division
        p_ndc = wp.vec3(p_clip[0] / p_clip[3], p_clip[1] / p_clip[3], p_clip[2] / p_clip[3])
        
        # 4. NDC to screen space
        p_screen = wp.vec2(
            (p_ndc[0] + 1.0) * 0.5 * float(image_width),
            (p_ndc[1] + 1.0) * 0.5 * float(image_height)
        )
        
        # Compute 3D covariance matrix
        # Construct scale matrix
        S = wp.mat33(
            scale[0], 0.0, 0.0,
            0.0, scale[1], 0.0,
            0.0, 0.0, scale[2]
        )
        
        # Construct rotation matrix from quaternion
        R = wp.mat33()
        R = wp.quat_to_matrix(quat)
        
        # Compute 3D covariance: Σ = R*S*S*R^T
        SS = wp.mat33(
            S[0, 0] * S[0, 0], 0.0, 0.0,
            0.0, S[1, 1] * S[1, 1], 0.0,
            0.0, 0.0, S[2, 2] * S[2, 2]
        )
        
        RS = wp.mul(R, SS)
        cov3d = wp.mul(RS, wp.transpose(R))
        
        # Compute 2D covariance matrix
        cov2d = compute_cov2d(cov3d, view_matrix, proj_matrix, tan_fovx, tan_fovy, float(image_width), float(image_height))
        
        # Check if the current pixel is within the Gaussian's influence
        # Compute Mahalanobis distance
        diff_x = pixel_pos[0] - p_screen[0]
        diff_y = pixel_pos[1] - p_screen[1]
        
        # Calculate determinant and inverse of covariance
        det = cov2d[0, 0] * cov2d[1, 1] - cov2d[0, 1] * cov2d[1, 0]
        if det <= 1e-10:
            continue
            
        inv_det = 1.0 / det
        
        inv_cov = wp.mat22(
            cov2d[1, 1] * inv_det, -cov2d[0, 1] * inv_det,
            -cov2d[1, 0] * inv_det, cov2d[0, 0] * inv_det
        )
        
        # Compute Mahalanobis distance squared
        dist_sq = diff_x * (inv_cov[0, 0] * diff_x + inv_cov[0, 1] * diff_y) + diff_y * (inv_cov[1, 0] * diff_x + inv_cov[1, 1] * diff_y)
        
        # Skip if pixel is outside the Gaussian's influence
        if dist_sq > 9.0:  # 3 sigma rule
            continue
            
        # Evaluate Gaussian
        gaussian_value = wp.exp(-0.5 * dist_sq) * opacity
        
        # Blend color based on alpha compositing (front-to-back)
        # Normalize depth to [0, 1]
        gaussian_depth = 0.5 * (p_ndc[2] + 1.0)
        
        # Only blend if this Gaussian is closer than accumulated depth so far
        if gaussian_depth < depth and gaussian_value > 0.001:
            # Alpha blending
            alpha = gaussian_value * (1.0 - accumulated_alpha)
            if alpha > 0.0:
                color = color + alpha * gaussian_color
                depth = gaussian_depth
                accumulated_alpha = accumulated_alpha + alpha
                
                # Early termination if opaque
                if accumulated_alpha >= 0.99:
                    break
    
    # Store final pixel values
    rendered_image[pixel_y, pixel_x] = color
    depth_image[pixel_y, pixel_x] = depth

def rasterize_gaussians(
    background,
    means3d,
    colors,
    opacities,
    scales,
    rotations,
    scale_modifier,
    view_matrix,
    proj_matrix,
    tan_fovx,
    tan_fovy,
    image_height,
    image_width,
    camera_pos,
    antialiasing=False
):
    """Wrapper function for the Gaussian rasterization kernel."""
    # Create output arrays
    rendered_image = wp.zeros((image_height, image_width), dtype=wp.vec3)
    depth_image = wp.zeros((image_height, image_width), dtype=float)
    
    # Convert inputs to Warp arrays if needed
    if not isinstance(means3d, wp.array):
        means3d_warp = wp.array(means3d, dtype=wp.vec3)
    else:
        means3d_warp = means3d
        
    if not isinstance(colors, wp.array):
        colors_warp = wp.array(colors, dtype=wp.vec3)
    else:
        colors_warp = colors
        
    if not isinstance(opacities, wp.array):
        opacities_warp = wp.array(opacities, dtype=float)
    else:
        opacities_warp = opacities
        
    if not isinstance(scales, wp.array):
        scales_warp = wp.array(scales, dtype=wp.vec3)
    else:
        scales_warp = scales
        
    if not isinstance(rotations, wp.array):
        rotations_warp = wp.array(rotations, dtype=wp.vec4)
    else:
        rotations_warp = rotations
    
    # Launch the kernel
    wp.launch(
        kernel=wp_rasterize_gaussians,
        dim=(image_width, image_height),
        inputs=[
            rendered_image,
            depth_image,
            wp.vec3(background[0], background[1], background[2]),
            means3d_warp,
            colors_warp,
            opacities_warp,
            scales_warp,
            rotations_warp,
            scale_modifier,
            view_matrix,
            proj_matrix,
            tan_fovx,
            tan_fovy,
            camera_pos,
            1 if antialiasing else 0
        ]
    )
    
    return rendered_image, depth_image
