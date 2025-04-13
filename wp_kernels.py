import numpy as np
import warp as wp
import math
from config import *
# Initialize Warp
wp.init()

# Added camera transformation functions
def wp_world_to_view(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def wp_projection_matrix(fovx, fovy, znear, zfar):
    tanHalfFovY = math.tan((fovy / 2))
    tanHalfFovX = math.tan((fovx / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def wp_preprocess(
    means3d,
    scales,
    rotations,
    opacities,
    shs,
    viewmatrix,
    projmatrix,
    cam_pos,
    W, H, focal_x, focal_y, tan_fovx, tan_fovy):
    pass

@wp.func
def compute_cov2d(p_orig: wp.vec3, cov3d: vec6, view_matrix: wp.mat44, 
                 tan_fovx: float, tan_fovy: float, width: float, height: float) -> wp.vec3:
    t = wp.transform_point(view_matrix, p_orig)
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    # Clamp X/Y to stay inside frustum
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = min(limx, max(-limx, txtz)) * t[2]
    t[1] = min(limy, max(-limy, tytz)) * t[2]
    focal_x = width / (2.0 * tan_fovx)
    focal_y = height / (2.0 * tan_fovy)
    # compute Jacobian
    J = wp.mat33(
        focal_x / t[2], 0.0, -(focal_x * t[0]) / (t[2] * t[2]),
        0.0, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2]),
        0.0, 0.0, 0.0
    )
    
    W = wp.mat33(
        view_matrix[0, 0], view_matrix[0, 1], view_matrix[0, 2],
        view_matrix[1, 0], view_matrix[1, 1], view_matrix[1, 2],
        view_matrix[2, 0], view_matrix[2, 1], view_matrix[2, 2]
    )
    T = J * W
    
    Vrk = wp.mat33(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    )
    
    cov = T * Vrk * wp.transpose(T)
    
    return wp.vec3(cov[0, 0], cov[0, 1], cov[1, 1])

@wp.func
def compute_cov3d(scale: wp.vec3, scale_mod: float, rot: wp.mat33) -> vec6:
    # Create scaling matrix with modifier applied
    S = wp.mat33(
        scale_mod * scale[0], 0.0, 0.0,
        0.0, scale_mod * scale[1], 0.0,
        0.0, 0.0, scale_mod * scale[2]
    )
    
    M = rot * S
    
    # Compute 3D covariance matrix: Sigma = M * M^T
    sigma = M * wp.transpose(M)
    
    return vec6(sigma[0, 0], sigma[0, 1], sigma[0, 2], sigma[1, 1], sigma[1, 2], sigma[2, 2])

@wp.kernel
def wp_preprocess(
    # Output buffers
    rendered_image: wp.array2d(dtype=wp.vec3),
    depth_image: wp.array2d(dtype=float),
    
    # Input parameters
    background: wp.vec3,
    pts: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
    opacities: wp.array(dtype=float),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.mat33),
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
    i = wp.tid()
    
    # Initialize pixel color and depth
    color = background
    depth = float(1.0)
    accumulated_alpha = float(0.0)
    
    # Get image dimensions
    image_height = rendered_image.shape[0]
    image_width = rendered_image.shape[1]
    # For each Gaussian
    p_orig = pts[i]
    # Transform to camera space
    p_view = wp.transform_point(view_matrix, p_orig)
    
    # project to clip space
    p_clip = proj_matrix * wp.vec4(p_orig[0], p_orig[1], p_orig[2], 1.0)
    p_ndc = wp.vec3(p_clip[0] / p_clip[3], p_clip[1] / p_clip[3], p_clip[2] / p_clip[3])
    
    p_screen = wp.vec2(
        (p_ndc[0] + 1.0) * 0.5 * float(image_width),
        (p_ndc[1] + 1.0) * 0.5 * float(image_height)
    )
    
    cov3d = compute_cov3d(scales[i], scale_modifier, rotations[i])
    
    # Compute 2D covariance matrix
    cov2d = compute_cov2d(p_orig, cov3d, view_matrix, tan_fovx, tan_fovy, float(image_width), float(image_height))

    #     # Check if the current pixel is within the Gaussian's influence
    #     # Compute Mahalanobis distance
    #     diff_x = pixel_pos[0] - p_screen[0]
    #     diff_y = pixel_pos[1] - p_screen[1]
        
    #     # Calculate determinant and inverse of covariance
    #     det = cov2d[0, 0] * cov2d[1, 1] - cov2d[0, 1] * cov2d[1, 0]
    #     if det <= 1e-10:
    #         continue
            
    #     inv_det = 1.0 / det
        
    #     inv_cov = wp.mat22(
    #         cov2d[1, 1] * inv_det, -cov2d[0, 1] * inv_det,
    #         -cov2d[1, 0] * inv_det, cov2d[0, 0] * inv_det
    #     )
        
    #     # Compute Mahalanobis distance squared
    #     dist_sq = diff_x * (inv_cov[0, 0] * diff_x + inv_cov[0, 1] * diff_y) + diff_y * (inv_cov[1, 0] * diff_x + inv_cov[1, 1] * diff_y)
        
    #     # Skip if pixel is outside the Gaussian's influence
    #     if dist_sq > 9.0:  # 3 sigma rule
    #         continue
            
    #     # Evaluate Gaussian
    #     gaussian_value = wp.exp(-0.5 * dist_sq) * opacity
        
    #     # Blend color based on alpha compositing (front-to-back)
    #     # Normalize depth to [0, 1]
    #     gaussian_depth = 0.5 * (p_ndc[2] + 1.0)
        
    #     # Only blend if this Gaussian is closer than accumulated depth so far
    #     if gaussian_depth < depth and gaussian_value > 0.001:
    #         # Alpha blending
    #         alpha = gaussian_value * (1.0 - accumulated_alpha)
    #         if alpha > 0.0:
    #             color = color + alpha * gaussian_color
    #             depth = gaussian_depth
    #             accumulated_alpha = accumulated_alpha + alpha
                
    #             # Early termination if opaque
    #             if accumulated_alpha >= 0.99:
    #                 break
    
    # # Store final pixel values
    # rendered_image[pixel_y, pixel_x] = color
    # depth_image[pixel_y, pixel_x] = depth

@wp.kernel
def wp_render_gaussians(
    # Output buffers
    rendered_image: wp.array2d(dtype=wp.vec3),
    depth_image: wp.array2d(dtype=float),
    
    # Input parameters
    background: wp.vec3,
    pts: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
    opacities: wp.array(dtype=float),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.mat33),
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
    pass

def render_gaussians(
    background,
    pts,
    colors,
    opacities,
    scales,
    rotations,
    scale_modifier,
    view_matrix,
    proj_matrix,
    tan_fovx,
    tan_fovy,
    sh_coeffs,
    image_height,
    image_width,
    camera_pos,
    antialiasing=False,
    verbose=True
):
    rendered_image = wp.zeros((image_height, image_width), dtype=wp.vec3)
    depth_image = wp.zeros((image_height, image_width), dtype=float)

    def to_warp_array(data, dtype, shape_check=None, flatten=False):
        if isinstance(data, wp.array):
            return data
        if flatten and data.ndim == 2 and data.shape[1] == 1:
            data = data.flatten()
        if shape_check:
            assert data.shape[1:] == shape_check, f"Expected shape {shape_check}, got {data.shape[1:]}"
        return wp.array(data, dtype=dtype)

    background_warp = wp.vec3(background[0], background[1], background[2])
    pts_warp = to_warp_array(pts, wp.vec3)
    colors_warp = wp.array(np.zeros((pts.shape[0], 3), dtype=np.float32), dtype=wp.vec3) if colors is None and sh_coeffs is not None else to_warp_array(colors, wp.vec3)
    opacities_warp = to_warp_array(opacities, float, flatten=True)
    scales_warp = to_warp_array(scales, wp.vec3)
    rotations_warp = to_warp_array(rotations, wp.mat33)
    
    view_matrix_warp = wp.mat44(view_matrix.flatten()) if not isinstance(view_matrix, wp.mat44) else view_matrix
    proj_matrix_warp = wp.mat44(proj_matrix.flatten()) if not isinstance(proj_matrix, wp.mat44) else proj_matrix
    camera_pos_warp = wp.vec3(camera_pos[0], camera_pos[1], camera_pos[2]) if not isinstance(camera_pos, wp.vec3) else camera_pos
    
    if verbose:
        print(f"\nWARP RENDERING: {image_width}x{image_height} image, {pts_warp.shape[0]} gaussians")
        print(f"Colors: {'from SH' if colors is None else 'provided'}, Rotations: {rotations_warp.shape[0]}")

    wp.launch(
        kernel=wp_preprocess,
        dim=pts_warp.shape[0],
        inputs=[
            rendered_image,
            depth_image,
            background_warp,
            pts_warp,
            colors_warp,
            opacities_warp,
            scales_warp,
            rotations_warp,
            scale_modifier,
            view_matrix_warp,
            proj_matrix_warp,
            tan_fovx,
            tan_fovy,
            camera_pos_warp,
            1 if antialiasing else 0
        ]
    )
    
    return rendered_image, depth_image

