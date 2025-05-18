import warp as wp
from utils.wp_utils import to_warp_array
from config import *
# Initialize Warp
wp.init()

# Define spherical harmonics constants
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199

@wp.func
def ndc2pix(x: float, size: float) -> float:
    return (x + 1.0) * 0.5 * size

@wp.func
def get_rect(p: wp.vec2, max_radius: float, tile_grid: wp.vec3):
    # Extract grid dimensions
    grid_size_x = tile_grid[0]
    grid_size_y = tile_grid[1]
    
    rect_min_x = wp.min(wp.int32(grid_size_x), wp.int32(wp.max(wp.int32(0), wp.int32((p[0] - max_radius) / float(TILE_M)))))
    rect_min_y = wp.min(wp.int32(grid_size_y), wp.int32(wp.max(wp.int32(0), wp.int32((p[1] - max_radius) / float(TILE_N)))))
    

    rect_max_x = wp.min(wp.int32(grid_size_x), wp.int32(wp.max(wp.int32(0), wp.int32((p[0] + max_radius + float(TILE_M) - 1.0) / float(TILE_M)))))
    rect_max_y = wp.min(wp.int32(grid_size_y), wp.int32(wp.max(wp.int32(0), wp.int32((p[1] + max_radius + float(TILE_N) - 1.0) / float(TILE_N)))))
    
    return rect_min_x, rect_min_y, rect_max_x, rect_max_y


@wp.func
def compute_cov2d(p_orig: wp.vec3, cov3d: VEC6, view_matrix: wp.mat44, 
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
    
    T = W * J
    
    Vrk = wp.mat33(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    )
    
    cov = T * Vrk * wp.transpose(T)
    
    return wp.vec3(cov[0, 0], cov[0, 1], cov[1, 1])

@wp.func
def compute_cov3d(scale: wp.vec3, scale_mod: float, rot: wp.vec4) -> VEC6:
    # Create scaling matrix with modifier applied
    S = wp.mat33(
        scale_mod * scale[0], 0.0, 0.0,
        0.0, scale_mod * scale[1], 0.0,
        0.0, 0.0, scale_mod * scale[2]
    )
    R = wp.quat_to_matrix(wp.quaternion(rot[0], rot[1], rot[2], rot[3]))
    M = R * S
    
    # Compute 3D covariance matrix: Sigma = M * M^T
    sigma = M * wp.transpose(M)
    
    return VEC6(sigma[0, 0], sigma[0, 1], sigma[0, 2], sigma[1, 1], sigma[1, 2], sigma[2, 2])

@wp.kernel
def wp_preprocess(
    orig_points: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    scale_modifier: float,
    rotations: wp.array(dtype=wp.vec4),
    
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
    cov3Ds: wp.array(dtype=VEC6),
    rgb: wp.array(dtype=wp.vec3),
    conic_opacity: wp.array(dtype=wp.vec4),
    tile_grid: wp.vec3,
    tiles_touched: wp.array(dtype=int),
    clamped_state: wp.array(dtype=wp.vec3),
    
    prefiltered: bool,
    antialiasing: bool
):
    # Get thread indices
    i = wp.tid()
    
    # For each Gaussian
    p_orig = orig_points[i]
    p_view = view_matrix * wp.vec4(p_orig[0], p_orig[1], p_orig[2], 1.0)

    if p_view[2] <= 0.2:
        return

    p_hom = wp.vec4(p_orig[0], p_orig[1], p_orig[2], 1.0) * proj_matrix
    
    p_w = 1.0 / (p_hom[3] + 0.0000001)
    p_proj = wp.vec3(p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w)

    cov3d = compute_cov3d(scales[i], scale_modifier, rotations[i])
    cov3Ds[i] = cov3d
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

    # Get rectangle of affected tiles
    rect_min_x, rect_min_y, rect_max_x, rect_max_y = get_rect(point_image, my_radius, tile_grid)
    # Skip if rectangle has 0 area
    if (rect_max_x - rect_min_x) * (rect_max_y - rect_min_y) == 0:
        return
    
    # Compute color from spherical harmonics
    pos = p_orig
    dir_orig = pos - cam_pos
    dir = wp.normalize(dir_orig)
    x, y, z = dir[0], dir[1], dir[2]
    
    # Base offset for this Gaussian's SH coefficients
    base_idx = i * 16  # assuming degree 3 (16 coefficients)
    
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
            
            # Degree 2 terms with hardcoded constants
            result = result + 1.0925484305920792 * xy * shs[base_idx + 4] 
            result = result + (-1.0925484305920792) * yz * shs[base_idx + 5]
            result = result + 0.31539156525252005 * (2.0 * zz - xx - yy) * shs[base_idx + 6]
            result = result + (-1.0925484305920792) * xz * shs[base_idx + 7]
            result = result + 0.5462742152960396 * (xx - yy) * shs[base_idx + 8]
                   
            if degree > 2:
                # Degree 3 terms with hardcoded constants
                result = result + (-0.5900435899266435) * y * (3.0 * xx - yy) * shs[base_idx + 9]
                result = result + 2.890611442640554 * xy * z * shs[base_idx + 10]
                result = result + (-0.4570457994644658) * y * (4.0 * zz - xx - yy) * shs[base_idx + 11]
                result = result + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shs[base_idx + 12]
                result = result + (-0.4570457994644658) * x * (4.0 * zz - xx - yy) * shs[base_idx + 13]
                result = result + 1.445305721320277 * z * (xx - yy) * shs[base_idx + 14]
                result = result + (-0.5900435899266435) * x * (xx - 3.0 * yy) * shs[base_idx + 15]
    
    result = result + wp.vec3(0.5, 0.5, 0.5)
    
    # Track which color channels are clamped (using wp.vec3 instead of separate uint32 values)
    # Store 1.0 if clamped, 0.0 if not clamped
    # Use separate assignments instead of conditional expressions
    r_clamped = 0.0
    g_clamped = 0.0
    b_clamped = 0.0
    
    if result[0] < 0.0:
        r_clamped = 1.0
    if result[1] < 0.0:
        g_clamped = 1.0
    if result[2] < 0.0:
        b_clamped = 1.0
        
    clamped_state[i] = wp.vec3(r_clamped, g_clamped, b_clamped)
    
    if clamped:
        # RGB colors are clamped to positive values
        result = wp.vec3(
            wp.max(result[0], 0.0),
            wp.max(result[1], 0.0),
            wp.max(result[2], 0.0)
        )

    rgb[i] = result
    
    # Store computed data
    depths[i] = p_view[2]
    radii[i] = int(my_radius)
    points_xy_image[i] = point_image
    
    # Pack conic and opacity into single vec4
    conic_opacity[i] = wp.vec4(conic[0], conic[1], conic[2], opacities[i])
    # Store tile information
    tiles_touched[i] = (rect_max_y - rect_min_y) * (rect_max_x - rect_min_x)

@wp.kernel
def wp_render_gaussians(
    # Output buffers
    rendered_image: wp.array2d(dtype=wp.vec3),
    depth_image: wp.array2d(dtype=float),
    
    # Tile data
    ranges: wp.array(dtype=wp.vec2i),
    point_list: wp.array(dtype=int),
    
    # Image parameters
    W: int,
    H: int,
    
    # Gaussian data
    points_xy_image: wp.array(dtype=wp.vec2),
    colors: wp.array(dtype=wp.vec3),
    conic_opacity: wp.array(dtype=wp.vec4),
    depths: wp.array(dtype=float),
    
    # Background color
    background: wp.vec3,
    
    # Tile grid info
    tile_grid: wp.vec3,
    
    # Track additional data
    final_Ts: wp.array2d(dtype=float),
    n_contrib: wp.array2d(dtype=int),
):
    tile_x, tile_y, tid_x, tid_y = wp.tid()
    
    # Calculate tile index
    
    if tile_y >= (H + TILE_N - 1) // TILE_N:
        return
    
    # Calculate pixel boundaries for this tile
    pix_min_x = tile_x * TILE_M
    pix_min_y = tile_y * TILE_N
    pix_max_x = wp.min(pix_min_x + TILE_M, W)
    pix_max_y = wp.min(pix_min_y + TILE_N, H)
    
    # Calculate pixel position for this thread
    pix_x = pix_min_x + tid_x
    pix_y = pix_min_y + tid_y
    
    # Check if this thread processes a valid pixel
    inside = (pix_x < W) and (pix_y < H)
    if not inside:
        return
    
    pixf_x = float(pix_x)
    pixf_y = float(pix_y)
    
    # Get start/end range of IDs to process for this tile
    tile_id = tile_y * int(tile_grid[0]) + tile_x
    range_start = ranges[tile_id][0]
    range_end = ranges[tile_id][1]
    
    # Initialize blending variables
    T = float(1.0)  # Transmittance
    r, g, b = float(0.0), float(0.0), float(0.0)  # Accumulated color
    expected_inv_depth = float(0.0)  # For depth calculation
    
    # Track the number of contributors to this pixel
    contributor_count = int(0)
    
    # Iterate over all Gaussians influencing this tile
    for i in range(range_start, range_end):
        # Get Gaussian ID
        gaussian_id = point_list[i]
        
        # Get Gaussian data
        xy = points_xy_image[gaussian_id]
        con_o = conic_opacity[gaussian_id]
        color = colors[gaussian_id]
        
        # Compute distance to Gaussian center
        d_x = xy[0] - pixf_x
        d_y = xy[1] - pixf_y
        
        # Compute Gaussian power (exponent)
        power = -0.5 * (con_o[0] * d_x * d_x + con_o[2] * d_y * d_y) - con_o[1] * d_x * d_y
        
        # Skip if power is positive (too far away)
        if power > 0.0:
            continue
        
        # Compute alpha from power and opacity
        alpha = wp.min(0.99, con_o[3] * wp.exp(power))
        
        # Skip if alpha is too small
        if alpha < (1.0 / 255.0):
            continue
        
        # Increment contributor count for this pixel
        contributor_count += 1
        
        # Test if we're close to fully opaque
        test_T = T * (1.0 - alpha)
        if test_T < 0.0001:
            break  # Early termination if pixel is almost opaque
        
        # Accumulate color contribution
        r += color[0] * alpha * T
        g += color[1] * alpha * T
        b += color[2] * alpha * T
        
        # Accumulate inverse depth
        expected_inv_depth += (1.0 / depths[gaussian_id]) * alpha * T
        
        # Update transmittance
        T = test_T
    
    # Store final transmittance (T) and contributor count
    final_Ts[pix_y, pix_x] = T
    n_contrib[pix_y, pix_x] = contributor_count
    
    # Write final color to output buffer (color + background)
    rendered_image[pix_y, pix_x] = wp.vec3(
        r + T * background[0],
        g + T * background[1],
        b + T * background[2]
    )
    
    # Write depth to output buffer
    depth_image[pix_y, pix_x] = expected_inv_depth

@wp.kernel
def wp_duplicate_with_keys(
    points_xy_image: wp.array(dtype=wp.vec2),
    depths: wp.array(dtype=float),
    point_offsets: wp.array(dtype=int),
    point_list_keys_unsorted: wp.array(dtype=wp.int64),
    point_list_unsorted: wp.array(dtype=int),
    radii: wp.array(dtype=int),
    tile_grid: wp.vec3
):
    tid = wp.tid()

    if tid >= points_xy_image.shape[0]:
        return

    r = radii[tid]
    if r <= 0:
        return

    # Find the global offset into key/value buffers
    offset = 0
    if tid > 0:
        offset = point_offsets[tid - 1]

    pos = points_xy_image[tid]
    depth_val = depths[tid]

    rect_min_x, rect_min_y, rect_max_x, rect_max_y = get_rect(pos, float(r), tile_grid)
    for y in range(rect_min_y, rect_max_y):
        for x in range(rect_min_x, rect_max_x):
            tile_id = y * int(tile_grid[0]) + x
            # Convert to int64 to avoid overflow during bit shift
            tile_id_64 = wp.int64(tile_id)
            shifted = tile_id_64 << wp.int64(32)
            # Combine tile ID and depth into single key
            key = wp.int64(shifted) | wp.int64(depth_val)

            point_list_keys_unsorted[offset] = key
            point_list_unsorted[offset] = tid
            offset += 1
            
@wp.kernel
def wp_identify_tile_ranges(
    num_rendered: int,
    point_list_keys: wp.array(dtype=wp.int64),
    ranges: wp.array(dtype=wp.vec2i)  # Each range is (start, end)
):
    idx = wp.tid()

    if idx >= num_rendered:
        return

    key = point_list_keys[idx]
    curr_tile = int(key >> wp.int64(32))

    # Set start of range if first element or tile changed
    if idx == 0:
        ranges[curr_tile][0] = 0
    else:
        prev_key = point_list_keys[idx - 1]
        prev_tile = int(prev_key >> wp.int64(32))
        if curr_tile != prev_tile:
            ranges[prev_tile][1] = idx
            ranges[curr_tile][0] = idx

    # Set end of range if last element
    if idx == num_rendered - 1:
        ranges[curr_tile][1] = num_rendered


@wp.kernel
def wp_prefix_sum(input_array: wp.array(dtype=int),
                      output_array: wp.array(dtype=int)):
    tid = wp.tid()
    
    if tid == 0:
        output_array[0] = input_array[0]
        
        # Perform prefix sum
        for i in range(1, input_array.shape[0]):
            output_array[i] = output_array[i-1] + input_array[i]


@wp.kernel
def wp_copy_int64(src: wp.array(dtype=wp.int64), dst: wp.array(dtype=wp.int64), count: int):
    i = wp.tid()
    if i < count:
        dst[i] = src[i]
        
@wp.kernel
def wp_copy_int(src: wp.array(dtype=int), dst: wp.array(dtype=int), count: int):
    i = wp.tid()
    if i < count:
        dst[i] = src[i]
        
@wp.kernel
def track_pixel_stats(
    rendered_image: wp.array2d(dtype=wp.vec3),
    depth_image: wp.array2d(dtype=float),
    background: wp.vec3,
    final_Ts: wp.array2d(dtype=float),
    n_contrib: wp.array2d(dtype=int),
    W: int,
    H: int
):
    """Kernel to track final transparency values and contributor counts for each pixel."""
    x, y = wp.tid()
    
    if x >= W or y >= H:
        return
    
    # Get the rendered pixel
    pixel = rendered_image[y, x]
    
    # Calculate approximate alpha transparency by checking for background contribution
    # If the pixel has no contribution from background, final_T should be close to 0
    # If it's mostly background, final_T will be close to 1
    diff_r = abs(pixel[0] - background[0])
    diff_g = abs(pixel[1] - background[1]) 
    diff_b = abs(pixel[2] - background[2])
    has_content = (diff_r > 0.01) or (diff_g > 0.01) or (diff_b > 0.01)
    
    if has_content:
        # Approximate final_T - in a real scenario this should already be tracked during rendering
        # We're just making sure it's populated for existing renderings
        if final_Ts[y, x] == 0.0:
            # If final_Ts hasn't been set during rendering, approximate it
            # Higher difference from background means lower T
            max_diff = max(diff_r, max(diff_g, diff_b))
            final_Ts[y, x] = 1.0 - min(0.99, max_diff)
        
        # Set n_contrib to 1 if we know the pixel has content but no contributor count
        if n_contrib[y, x] == 0:
            n_contrib[y, x] = 1

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
    clamped=True,
    debug=False,
):
    """Render 3D Gaussians using Warp.
    
    Args:
        background: Background color tensor of shape (3,)
        means3D: 3D positions tensor of shape (N, 3)
        colors: Optional RGB colors tensor of shape (N, 3)
        opacity: Opacity values tensor of shape (N, 1) or (N,)
        scales: Scales tensor of shape (N, 3)
        rotations: Rotation quaternions of shape (N, 4)
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
        clamped: Whether to clamp the colors
        debug: Whether to print debug information
        
    Returns:
        Tuple of (rendered_image, depth_image, intermediate_buffers)
    """
    rendered_image = wp.zeros((image_height, image_width), dtype=wp.vec3, device=DEVICE)
    depth_image = wp.zeros((image_height, image_width), dtype=float, device=DEVICE)
    
    # Create additional buffers for tracking transparency and contributors
    final_Ts = wp.zeros((image_height, image_width), dtype=float, device=DEVICE)
    n_contrib = wp.zeros((image_height, image_width), dtype=int, device=DEVICE)

    background_warp = wp.vec3(background[0], background[1], background[2])
    points_warp = to_warp_array(means3D, wp.vec3)
    

    # SH coefficients should be shape (n, 16, 3)
    # Convert to a flattened array but preserve the structure
    sh_data = sh.reshape(-1, 3) if hasattr(sh, 'reshape') else sh
    shs_warp = to_warp_array(sh_data, wp.vec3)
    
    # Handle other parameters
    opacities_warp = to_warp_array(opacity, float, flatten=True)
    scales_warp = to_warp_array(scales, wp.vec3)
    rotations_warp = to_warp_array(rotations, wp.vec4)

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
    radii = wp.zeros(num_points, dtype=int, device=DEVICE)
    points_xy_image = wp.zeros(num_points, dtype=wp.vec2, device=DEVICE)
    depths = wp.zeros(num_points, dtype=float, device=DEVICE)
    cov3Ds = wp.zeros(num_points, dtype=VEC6, device=DEVICE)
    rgb = wp.zeros(num_points, dtype=wp.vec3, device=DEVICE)
    conic_opacity = wp.zeros(num_points, dtype=wp.vec4, device=DEVICE)
    tiles_touched = wp.zeros(num_points, dtype=int, device=DEVICE)
    
    # Add clamped_state buffer to track which color channels are clamped
    clamped_state = wp.zeros(num_points, dtype=wp.vec3, device=DEVICE)
    
    if debug:
        print(f"\nWARP RENDERING: {image_width}x{image_height} image, {num_points} gaussians")
        print(f"Colors: {'from SH' if colors is None else 'provided'}, SH degree: {degree}")
        print(f"Antialiasing: {antialiasing}, Prefiltered: {prefiltered}")

    # Launch preprocessing kernel
    wp.launch(
        kernel=wp_preprocess,
        dim=(num_points,),
        inputs=[
            points_warp,               # orig_points
            scales_warp,               # scales
            scale_modifier,            # scale_modifier
            rotations_warp,            # rotations_quat
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
            clamped_state,             # clamped_state - now using wp.vec3
            prefiltered,               # prefiltered
            antialiasing               # antialiasing
        ],
    )
    
    torch_points_xy_image = wp.to_torch(points_xy_image).cpu().numpy()
    print("torch_points_xy_image", torch_points_xy_image.shape, torch_points_xy_image.flatten()[:100])
    # print how many has values
    print("number of points with values", np.sum(torch_points_xy_image[:, 0] != 0.0))
    point_offsets = wp.zeros(num_points, dtype=int, device=DEVICE)
    wp.launch(
        kernel=wp_prefix_sum,
        dim=1,
        inputs=[
            tiles_touched,
            point_offsets
        ]
    )

    num_rendered = int(wp.to_torch(point_offsets)[-1].item())  # total number of duplicated entries
    
    if num_rendered > (1 << 30):
        # radix sort needs 2x memory
        raise ValueError("Number of rendered points exceeds the maximum supported by Warp.")
    
    point_list_keys_unsorted = wp.zeros(num_rendered, dtype=wp.int64, device=DEVICE)
    point_list_unsorted = wp.zeros(num_rendered, dtype=int, device=DEVICE)
    point_list_keys = wp.zeros(num_rendered, dtype=wp.int64, device=DEVICE)
    point_list = wp.zeros(num_rendered, dtype=int, device=DEVICE)
    wp.launch(
        kernel=wp_duplicate_with_keys,
        dim=num_points,
        inputs=[
            points_xy_image,
            depths,
            point_offsets,
            point_list_keys_unsorted,
            point_list_unsorted,
            radii,
            tile_grid
        ]
    )

    point_list_keys_unsorted_padded = wp.zeros(num_rendered * 2, dtype=wp.int64, device=DEVICE) 
    point_list_unsorted_padded = wp.zeros(num_rendered * 2, dtype=int, device=DEVICE)
    
    # Copy data to padded arrays
    wp.copy(point_list_keys_unsorted_padded, point_list_keys_unsorted)
    wp.copy(point_list_unsorted_padded, point_list_unsorted)
    wp.utils.radix_sort_pairs(
        point_list_keys_unsorted_padded,  # keys to sort
        point_list_unsorted_padded,       # values to sort along with keys
        num_rendered                      # number of elements to sort
    )

    wp.launch(
        kernel=wp_copy_int64,
        dim=num_rendered,
        inputs=[
            point_list_keys_unsorted_padded,
            point_list_keys,
            num_rendered
        ]
    )
    
    wp.launch(
        kernel=wp_copy_int,
        dim=num_rendered, 
        inputs=[
            point_list_unsorted_padded,
            point_list,
            num_rendered
        ]
    )
    
    tile_count = int(tile_grid[0] * tile_grid[1])
    ranges = wp.zeros(tile_count, dtype=wp.vec2i, device=DEVICE)  # each is (start, end)

    if num_rendered > 0:
        wp.launch(
            kernel=wp_identify_tile_ranges,  # You also need this kernel
            dim=num_rendered,
            inputs=[
                num_rendered,
                point_list_keys,
                ranges
            ]
        )
        
        num_tiles = tile_count
        
        wp.launch(
            kernel=wp_render_gaussians,
            dim=(int(tile_grid[0]), int(tile_grid[1]), TILE_M, TILE_N),
            inputs=[
                rendered_image,        # Output color image
                depth_image,           # Output depth image
                ranges,                # Tile ranges
                point_list,            # Sorted point indices
                image_width,           # Image width
                image_height,          # Image height
                points_xy_image,       # 2D points
                rgb,                   # Precomputed colors
                conic_opacity,         # Conic matrices and opacities
                depths,                # Depth values
                background_warp,       # Background color
                tile_grid,             # Tile grid configuration
                final_Ts,              # Final transparency values
                n_contrib,             # Number of contributors per pixel
            ]
        )
        
        # Launch the pixel stats tracking kernel as a fallback
        # to make sure final_Ts and n_contrib are populated
        # This is especially important for existing rendered pixels
        wp.launch(
            kernel=track_pixel_stats,
            dim=(image_width, image_height),
            inputs=[
                rendered_image,
                depth_image,
                background_warp,
                final_Ts,
                n_contrib,
                image_width,
                image_height
            ]
        )

    # print("radii", radii.shape, radii.flatten()[:100])
    # Return rendered image, depth image, and intermediate buffers needed for backward pass
    return rendered_image, depth_image, {
        "radii": radii,
        "point_offsets": point_offsets,
        "points_xy_image": points_xy_image,
        "depths": depths,
        "rgb": rgb,
        "cov3Ds": cov3Ds,
        "conic_opacity": conic_opacity,
        "point_list": point_list,
        "ranges": ranges,
        "final_Ts": final_Ts,  # Add final_Ts to intermediate buffers
        "n_contrib": n_contrib,  # Add contributor count to intermediate buffers
        "clamped_state": clamped_state  # Add clamped state to intermediate buffers
    }