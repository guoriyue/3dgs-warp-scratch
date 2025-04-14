import numpy as np
import warp as wp
import math
from config import *
# Initialize Warp
wp.init()



@wp.func
def ndc2pix(x: float, size: float) -> float:
    return (x + 1.0) * 0.5 * size

@wp.func
def get_rect(p: wp.vec2, max_radius: float, tile_grid: wp.vec3):
    # Extract grid dimensions
    grid_size_x = tile_grid[0]
    grid_size_y = tile_grid[1]
    
    # Calculate rectangle bounds matching CUDA implementation
    rect_min = wp.vec2(
        wp.min(grid_size_x, wp.max(0, wp.int32((p[0] - max_radius) / TILE_M))),
        wp.min(grid_size_y, wp.max(0, wp.int32((p[1] - max_radius) / TILE_N)))
    )
    
    rect_max = wp.vec2(
        wp.min(grid_size_x, wp.max(0, wp.int32((p[0] + max_radius + TILE_M - 1.0) / TILE_M))),
        wp.min(grid_size_y, wp.max(0, wp.int32((p[1] + max_radius + TILE_N - 1.0) / TILE_N)))
    )
    
    return rect_min, rect_max


@wp.func
def compute_color_from_sh(
    i: int,
    points: wp.array(dtype=wp.vec3),
    cam_pos: wp.vec3,
    shs: wp.array(dtype=wp.vec3),
    clamped: bool
) -> wp.vec3:
    # Note: Implementation would depend on your spherical harmonics setup
    # This is a placeholder
    view_dir = wp.normalize(cam_pos - points[i])
    
    # Simple placeholder for SH lighting (would need to be replaced with actual implementation)
    result = wp.vec3(shs[i])
    
    if clamped:
        result = wp.vec3(
            wp.clamp(result[0], 0.0, 1.0),
            wp.clamp(result[1], 0.0, 1.0),
            wp.clamp(result[2], 0.0, 1.0)
        )
        
    return result

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


@wp.func
def in_frustum(p_orig: wp.vec3, view_matrix: wp.mat44):
    # bring point to screen space
    p_view = wp.transform_point(view_matrix, p_orig)

    if p_view[2] <= 0.2:
        return None
    return p_view

@wp.kernel
def wp_preprocess(
    orig_points: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    scale_modifier: float,
    rotations: wp.array(dtype=wp.mat33),
    
    opacities: wp.array(dtype=float),
    shs: wp.array(dtype=wp.vec3),
    clamped: wp.array(dtype=bool),
    
    cov3D_precomp: wp.array(dtype=vec6),
    colors_precomp: wp.array(dtype=wp.vec3),
    
    view_matrix: wp.mat44,
    proj_matrix: wp.mat44,
    cam_pos: wp.vec3,
    
    W: int,
    H: int,
    
    tan_fovx: float,
    tan_fovy: float,
    
    focal_x: float,
    focal_y: float,
    
    radii: wp.array(dtype=int),
    points_xy_image: wp.array(dtype=wp.vec2),
    depths: wp.array(dtype=float),
    cov3Ds: wp.array(dtype=vec6),
    rgb: wp.array(dtype=wp.vec3),
    conic_opacity: wp.array(dtype=wp.vec4),
    tile_grid: wp.vec3,
    tiles_touched: wp.array(dtype=int),
    
    prefiltered: bool,
    antialiasing: bool
):
    # Get thread indices
    i = wp.tid()
    
    # For each Gaussian
    p_orig = orig_points[i]
    
    p_view = in_frustum(p_orig, view_matrix)
    if p_view is None:
        return
    
    p_hom = wp.vec4(p_orig[0], p_orig[1], p_orig[2], 1.0)
    p_hom = proj_matrix * p_hom
    p_w = 1.0 / (p_hom[3] + 0.0000001)
    p_proj = wp.vec3(p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w)
    
    cov3d = compute_cov3d(scales[i], scale_modifier, rotations[i])
    
    # Compute 2D covariance matrix
    cov2d = compute_cov2d(p_orig, cov3d, view_matrix, tan_fovx, tan_fovy, float(image_width), float(image_height))
    
    
    # Constants
    h_var = 0.3
    W = float(W)
    H = float(H)
    C = 3  # RGB channels
    
    # Add blur/antialiasing factor to covariance
    det_cov = cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1]
    cov_with_blur = wp.vec3(cov2d[0] + h_var, cov2d[1], cov2d[2] + h_var)
    det_cov_plus_h_cov = cov_with_blur[0] * cov_with_blur[2] - cov_with_blur[1] * cov_with_blur[1]
    
    # Antialiasing scaling factor
    h_convolution_scaling = 1.0
    if antialiasing:
        h_convolution_scaling = wp.sqrt(wp.max(0.000025, det_cov / det_cov_plus_h_cov))
    
    # Invert covariance (EWA algorithm)
    det = det_cov_plus_h_cov
    
    if det == 0.0:
        return
        
    det_inv = 1.0 / det
    conic = wp.vec3(
        cov_with_blur[2] * det_inv, 
        -cov_with_blur[1] * det_inv, 
        cov_with_blur[0] * det_inv
    )
    
    # Compute eigenvalues of covariance matrix to find screen-space extent
    mid = 0.5 * (cov_with_blur[0] + cov_with_blur[2])
    lambda1 = mid + wp.sqrt(wp.max(0.1, mid * mid - det))
    lambda2 = mid - wp.sqrt(wp.max(0.1, mid * mid - det))
    my_radius = wp.ceil(3.0 * wp.sqrt(wp.max(lambda1, lambda2)))
    
    # Convert to pixel coordinates
    point_image = wp.vec2(ndc2pix(p_proj[0], W), ndc2pix(p_proj[1], H))
    # grid = wp.vec2(image_width // TILE_M, image_height // TILE_N)
    # Get rectangle of affected tiles
    rect_min, rect_max = get_rect(point_image, my_radius, tile_grid)
    
    # Skip if rectangle has 0 area
    if (rect_max[0] - rect_min[0]) * (rect_max[1] - rect_min[1]) == 0:
        return
    
    # Process colors
    if colors_precomp is None:
        # Compute color from spherical harmonics
        result = compute_color_from_sh(i, orig_points, cam_pos, shs, clamped)
        rgb[i * 3 + 0] = result[0]
        rgb[i * 3 + 1] = result[1]
        rgb[i * 3 + 2] = result[2]
    
    # Store computed data
    depths[i] = p_view[2]
    radii[i] = my_radius
    points_xy_image[i] = point_image
    
    # Pack conic and opacity into single vec4
    opacity = opacities[i]
    conic_opacity[i] = wp.vec4(conic[0], conic[1], conic[2], opacity * h_convolution_scaling)
    
    # Store tile information
    tiles_touched[i] = (rect_max[1] - rect_min[1]) * (rect_max[0] - rect_min[0])

@wp.kernel
def wp_render_gaussians(
    # Output buffers
    rendered_image: wp.array2d(dtype=wp.vec3),
    depth_image: wp.array2d(dtype=float),
    
    # Input parameters
    background: wp.vec3,
    orig_points: wp.array(dtype=wp.vec3),
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
    orig_points,
    scales,
    scale_modifier,
    rotation,
    opacities,
    shs,
    clamped,
    cov3D_precomp,
    colors_precomp,
    
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
    orig_points_warp = to_warp_array(orig_points, wp.vec3)
    colors_warp = wp.array(np.zeros((orig_points.shape[0], 3), dtype=np.float32), dtype=wp.vec3) if colors is None and sh_coeffs is not None else to_warp_array(colors, wp.vec3)
    opacities_warp = to_warp_array(opacities, float, flatten=True)
    scales_warp = to_warp_array(scales, wp.vec3)
    rotations_warp = to_warp_array(rotations, wp.mat33)
    
    view_matrix_warp = wp.mat44(view_matrix.flatten()) if not isinstance(view_matrix, wp.mat44) else view_matrix
    proj_matrix_warp = wp.mat44(proj_matrix.flatten()) if not isinstance(proj_matrix, wp.mat44) else proj_matrix
    camera_pos_warp = wp.vec3(camera_pos[0], camera_pos[1], camera_pos[2]) if not isinstance(camera_pos, wp.vec3) else camera_pos
    
    tile_grid = wp.vec3((image_width + TILE_M - 1) // TILE_M, (image_height + TILE_N - 1) // TILE_N, 1)
    
    if verbose:
        print(f"\nWARP RENDERING: {image_width}x{image_height} image, {orig_points_warp.shape[0]} gaussians")
        print(f"Colors: {'from SH' if colors is None else 'provided'}, Rotations: {rotations_warp.shape[0]}")

    wp.launch(
        kernel=wp_preprocess,
        dim=orig_points_warp.shape[0],
        inputs=[
            rendered_image,
            depth_image,
            background_warp,
            orig_points_warp,
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
            tile_grid,
            1 if antialiasing else 0
        ]
    )
    
    return rendered_image, depth_image
