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
    print(grid_size_x)
    print(wp.max(0, wp.int32((p[0] - max_radius) / float(TILE_M))))
    # Calculate rectangle bounds matching CUDA implementation
    rect_min_x = wp.max(wp.int32(0), wp.int32((p[0] - max_radius) / float(TILE_M)))
    rect_min_y = wp.max(wp.int32(0), wp.int32((p[1] - max_radius) / float(TILE_N)))
    

    rect_max_x = wp.max(wp.int32(0), wp.int32((p[0] + max_radius + float(TILE_M) - 1.0) / float(TILE_M)))
    rect_max_y = wp.max(wp.int32(0), wp.int32((p[1] + max_radius + float(TILE_N) - 1.0) / float(TILE_N)))
    
    return rect_min_x, rect_min_y, rect_max_x, rect_max_y


@wp.func
def compute_color_from_sh(
    idx: int,
    points: wp.array(dtype=wp.vec3),
    campos: wp.vec3,
    shs: wp.array(dtype=wp.vec3),
    degree: int,
    clamped: bool
) -> wp.vec3:
    """Compute colors from spherical harmonics coefficients.
    
    Args:
        idx: Index of the point (Gaussian)
        points: Array of 3D positions
        campos: Camera position
        shs: Array of SH coefficients, flattened from (n, 16, 3) to (n*16, 3)
        degree: Degree of SH to compute (0, 1, 2, or 3)
        clamped: Whether to clamp colors to [0,1]
        
    Returns:
        RGB color as vec3
    """
    # Constants for spherical harmonics (copied from render_python/sh.py)
    SH_C0 = 0.28209479177387814
    SH_C1 = 0.4886025119029199
    
    # Calculate view direction
    pos = points[idx]
    dir = pos - campos
    dir = wp.normalize(dir)
    x, y, z = dir[0], dir[1], dir[2]
    
    # Base offset for this Gaussian's SH coefficients
    base_idx = idx * 16  # assuming degree 3 (16 coefficients)
    
    # Start with the DC component (degree 0)
    result = SH_C0 * shs[base_idx]
    
    # Add higher degree terms if requested
    if degree > 0:
        # Degree 1 terms
        result = result - SH_C1 * y * shs[base_idx + 1] + SH_C1 * z * shs[base_idx + 2] - SH_C1 * x * shs[base_idx + 3]
        
        if degree > 1:
            # Degree 2 terms
            xx = x*x
            yy = y*y
            zz = z*z
            xy = x*y
            yz = y*z
            xz = x*z
            
            # Using exact same constants as in render_python/sh.py
            result = result + 1.0925484305920792 * xy * shs[base_idx + 4] \
                   + (-1.0925484305920792) * yz * shs[base_idx + 5] \
                   + 0.31539156525252005 * (2.0 * zz - xx - yy) * shs[base_idx + 6] \
                   + (-1.0925484305920792) * xz * shs[base_idx + 7] \
                   + 0.5462742152960396 * (xx - yy) * shs[base_idx + 8]
                   
            if degree > 2:
                # Degree 3 terms using exact same constants as in render_python/sh.py
                result = result \
                       + (-0.5900435899266435) * y * (3.0 * xx - yy) * shs[base_idx + 9] \
                       + 2.890611442640554 * xy * z * shs[base_idx + 10] \
                       + (-0.4570457994644658) * y * (4.0 * zz - xx - yy) * shs[base_idx + 11] \
                       + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shs[base_idx + 12] \
                       + (-0.4570457994644658) * x * (4.0 * zz - xx - yy) * shs[base_idx + 13] \
                       + 1.445305721320277 * z * (xx - yy) * shs[base_idx + 14] \
                       + (-0.5900435899266435) * x * (xx - 3.0 * yy) * shs[base_idx + 15]
    
    result = result + wp.vec3(0.5, 0.5, 0.5)
    
    if clamped:
        # RGB colors are clamped to positive values
        result = wp.vec3(
            wp.max(result[0], 0.0),
            wp.max(result[1], 0.0),
            wp.max(result[2], 0.0)
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

    return p_view

@wp.kernel
def wp_preprocess(
    orig_points: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    scale_modifier: float,
    rotations: wp.array(dtype=wp.mat33),
    
    opacities: wp.array(dtype=float),
    shs: wp.array(dtype=wp.vec3),
    degree: int,
    clamped: bool,
    
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

    if p_view[2] <= 0.2:
        return
    
    p_hom = wp.vec4(p_orig[0], p_orig[1], p_orig[2], 1.0)
    p_hom = proj_matrix * p_hom
    p_w = 1.0 / (p_hom[3] + 0.0000001)
    p_proj = wp.vec3(p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w)
    
    cov3d = compute_cov3d(scales[i], scale_modifier, rotations[i])
    
    # Compute 2D covariance matrix
    cov2d = compute_cov2d(p_orig, cov3d, view_matrix, tan_fovx, tan_fovy, float(W), float(H))
    
    
    # Constants
    h_var = 0.3
    W_float = float(W)
    H_float = float(H)
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
    point_image = wp.vec2(ndc2pix(p_proj[0], W_float), ndc2pix(p_proj[1], H_float))
    # grid = wp.vec2(image_width // TILE_M, image_height // TILE_N)
    # Get rectangle of affected tiles
    rect_min_x, rect_min_y, rect_max_x, rect_max_y = get_rect(point_image, my_radius, tile_grid)
    
    # Skip if rectangle has 0 area
    if (rect_max_x - rect_min_x) * (rect_max_y - rect_min_y) == 0:
        return
    
    # Compute color from spherical harmonics
    result = compute_color_from_sh(i, orig_points, cam_pos, shs, 3, clamped)
    
    rgb[i] = result
    
    # Store computed data
    depths[i] = p_view[2]
    radii[i] = int(my_radius)
    points_xy_image[i] = point_image
    
    # Pack conic and opacity into single vec4
    opacity = opacities[i]
    conic_opacity[i] = wp.vec4(conic[0], conic[1], conic[2], opacity * h_convolution_scaling)
    
    # Store tile information
    tiles_touched[i] = (rect_max_y - rect_min_y) * (rect_max_x - rect_min_x)

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
    means3D,
    colors=None,
    opacity=None,
    scales=None,
    rotations=None,
    scale_modifier=1.0,
    viewmatrix=None,
    projmatrix=None,
    tan_fovx=0.5, 
    tan_fovy=0.5,
    image_height=256,
    image_width=256,
    sh=None,
    degree=3,
    campos=None,
    prefiltered=False,
    antialiasing=False,
    debug=False
):
    """Render 3D Gaussians using Warp.
    
    Args:
        background: Background color tensor of shape (3,)
        means3D: 3D positions tensor of shape (N, 3)
        colors: Optional RGB colors tensor of shape (N, 3)
        opacity: Opacity values tensor of shape (N, 1) or (N,)
        scales: Scales tensor of shape (N, 3)
        rotations: Rotation matrices tensor of shape (N, 3, 3)
        scale_modifier: Global scale modifier (float)
        viewmatrix: View matrix tensor of shape (4, 4)
        projmatrix: Projection matrix tensor of shape (4, 4)
        tan_fovx: Tangent of the horizontal field of view
        tan_fovy: Tangent of the vertical field of view
        image_height: Height of the output image
        image_width: Width of the output image
        sh: Spherical harmonics coefficients tensor of shape (N, D, 3)
        degree: Degree of spherical harmonics
        campos: Camera position tensor of shape (3,)
        prefiltered: Whether input Gaussians are prefiltered
        antialiasing: Whether to apply antialiasing
        debug: Whether to print debug information
        
    Returns:
        Tuple of (rendered_image, depth_image) as Warp arrays
    """
    rendered_image = wp.zeros((image_height, image_width), dtype=wp.vec3)
    depth_image = wp.zeros((image_height, image_width), dtype=float)

    def to_warp_array(data, dtype, shape_check=None, flatten=False):
        if isinstance(data, wp.array):
            return data
        if data is None:
            return None
        # Convert torch tensor to numpy if needed
        if hasattr(data, 'cpu') and hasattr(data, 'numpy'):
            data = data.cpu().numpy()
        if flatten and data.ndim == 2 and data.shape[1] == 1:
            data = data.flatten()
        if shape_check and data.shape[1:] != shape_check:
            if debug:
                print(f"Warning: Expected shape {shape_check}, got {data.shape[1:]}")
        return wp.array(data, dtype=dtype)

    background_warp = wp.vec3(background[0], background[1], background[2])
    points_warp = to_warp_array(means3D, wp.vec3)
    

    # SH coefficients should be shape (n, 16, 3)
    # Convert to a flattened array but preserve the structure
    sh_data = sh.reshape(-1, 3) if hasattr(sh, 'reshape') else sh
    shs_warp = to_warp_array(sh_data, wp.vec3)
    
    # Handle other parameters
    opacities_warp = to_warp_array(opacity, float, flatten=True)
    scales_warp = to_warp_array(scales, wp.vec3)
    rotations_warp = to_warp_array(rotations, wp.mat33)

    # Handle camera parameters
    view_matrix_warp = wp.mat44(viewmatrix.flatten()) if not isinstance(viewmatrix, wp.mat44) else viewmatrix
    proj_matrix_warp = wp.mat44(projmatrix.flatten()) if not isinstance(projmatrix, wp.mat44) else projmatrix
    campos_warp = wp.vec3(campos[0], campos[1], campos[2]) if not isinstance(campos, wp.vec3) else campos
    
    # Calculate tile grid for spatial optimization
    tile_grid = wp.vec3((image_width + TILE_M - 1) // TILE_M, 
                        (image_height + TILE_N - 1) // TILE_N, 
                        1)
    
    # Preallocate buffers for preprocessed data
    num_points = points_warp.shape[0]
    radii = wp.zeros(num_points, dtype=int)
    points_xy_image = wp.zeros(num_points, dtype=wp.vec2)
    depths = wp.zeros(num_points, dtype=float)
    cov3Ds = wp.zeros(num_points, dtype=vec6)
    rgb = wp.zeros(num_points, dtype=wp.vec3)
    conic_opacity = wp.zeros(num_points, dtype=wp.vec4)
    tiles_touched = wp.zeros(num_points, dtype=int)
    
    clamped = True  # Default to clamped colors
    
    if debug:
        print(f"\nWARP RENDERING: {image_width}x{image_height} image, {num_points} gaussians")
        print(f"Colors: {'from SH' if colors is None else 'provided'}, SH degree: {degree}")
        print(f"Antialiasing: {antialiasing}, Prefiltered: {prefiltered}")

    # Launch preprocessing kernel
    wp.launch(
        kernel=wp_preprocess,
        dim=num_points,
        inputs=[
            points_warp,               # orig_points
            scales_warp,               # scales
            scale_modifier,            # scale_modifier
            rotations_warp,            # rotations
            opacities_warp,            # opacities
            shs_warp,                  # shs
            degree,
            clamped,                   # clamped
            view_matrix_warp,          # view_matrix
            proj_matrix_warp,          # proj_matrix
            campos_warp,               # cam_pos
            image_width,               # W
            image_height,              # H
            tan_fovx,                  # tan_fovx
            tan_fovy,                  # tan_fovy
            image_width / (2.0 * tan_fovx),  # focal_x
            image_height / (2.0 * tan_fovy),  # focal_y
            radii,                     # radii
            points_xy_image,           # points_xy_image
            depths,                    # depths
            cov3Ds,                    # cov3Ds
            rgb,                       # rgb
            conic_opacity,             # conic_opacity
            tile_grid,                 # tile_grid
            tiles_touched,             # tiles_touched
            prefiltered,               # prefiltered
            antialiasing               # antialiasing
        ]
    )
    
    return rendered_image, depth_image
