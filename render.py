import numpy as np
import warp as wp
import matplotlib.pyplot as plt
from wp_kernels import rasterize_gaussians
import imageio
from config import *

# Initialize Warp
wp.init()

def compute_sh_color(degrees, position, cam_pos, sh_coeffs):
    """Compute color from spherical harmonics coefficients."""
    # Constants for SH calculations
    SH_C0 = 0.28209479177387814
    SH_C1 = 0.4886025119029199
    SH_C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    ]
    SH_C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    ]
    
    # Compute view direction
    dir = position - cam_pos
    dir = dir / np.linalg.norm(dir)
    x, y, z = dir
    
    # Compute color from SH coefficients
    result = SH_C0 * sh_coeffs[0]
    
    if degrees > 0:
        result = result - SH_C1 * y * sh_coeffs[1] + SH_C1 * z * sh_coeffs[2] - SH_C1 * x * sh_coeffs[3]

        if degrees > 1:
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z
            result = (
                result
                + SH_C2[0] * xy * sh_coeffs[4]
                + SH_C2[1] * yz * sh_coeffs[5]
                + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[6]
                + SH_C2[3] * xz * sh_coeffs[7]
                + SH_C2[4] * (xx - yy) * sh_coeffs[8]
            )

            if degrees > 2:
                result = (
                    result
                    + SH_C3[0] * y * (3.0 * xx - yy) * sh_coeffs[9]
                    + SH_C3[1] * xy * z * sh_coeffs[10]
                    + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coeffs[11]
                    + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeffs[12]
                    + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coeffs[13]
                    + SH_C3[5] * z * (xx - yy) * sh_coeffs[14]
                    + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coeffs[15]
                )
    
    result += 0.5
    return np.clip(result, 0.0, 1.0)

def transform_point(point, matrix):
    """Transform a 3D point using a 4x4 matrix."""
    p = np.append(point, 1.0)  # Homogeneous coordinates
    result = np.dot(matrix, p)
    if result[3] != 0:
        result = result / result[3]  # Perspective division
    return result[:3]

def in_frustum(point, view_matrix):
    """Check if a point is in the camera frustum."""
    # Transform point to camera space
    p_view = transform_point(point, view_matrix)
    
    # Check if point is in front of camera
    if p_view[2] <= 0.2:  # Near plane
        return None
    
    return p_view

def create_camera_matrices(position, look_at, up, fov_degrees, aspect_ratio):
    """Create view and projection matrices for a camera."""
    # View matrix (look-at matrix)
    forward = np.array(look_at) - np.array(position)
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, np.array(up))
    right = right / np.linalg.norm(right)
    
    new_up = np.cross(right, forward)
    
    # Convert to column-major format for warp
    view_matrix = np.zeros((4, 4), dtype=np.float32)
    view_matrix[0, 0:3] = right
    view_matrix[1, 0:3] = new_up
    view_matrix[2, 0:3] = -forward  # Camera points in -z direction
    view_matrix[0, 3] = -np.dot(right, position)
    view_matrix[1, 3] = -np.dot(new_up, position)
    view_matrix[2, 3] = np.dot(forward, position)
    view_matrix[3, 3] = 1.0
    
    # Projection matrix (perspective)
    fov_radians = np.radians(fov_degrees)
    f = 1.0 / np.tan(fov_radians / 2.0)
    
    projection_matrix = np.zeros((4, 4), dtype=np.float32)
    projection_matrix[0, 0] = f / aspect_ratio
    projection_matrix[1, 1] = f
    projection_matrix[2, 2] = -1.0  # Far plane at infinity
    projection_matrix[2, 3] = -0.1  # Near plane
    projection_matrix[3, 2] = -1.0
    
    return wp.mat44(view_matrix.flatten()), wp.mat44(projection_matrix.flatten()), view_matrix, projection_matrix

def create_sample_gaussians(num_gaussians=50, with_sh=False, sh_degree=3):
    """Create sample 3D Gaussians."""
    # Random positions in a sphere
    radius = 3.0
    theta = np.random.uniform(0, 2 * np.pi, num_gaussians)
    phi = np.random.uniform(0, np.pi, num_gaussians)
    r = radius * np.random.uniform(0.5, 1.0, num_gaussians)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    means = np.column_stack((x, y, z)).astype(np.float32)
    
    # Random colors or SH coefficients
    if with_sh:
        # Number of SH coefficients based on degree
        num_sh_coeffs = (sh_degree + 1) ** 2
        sh_coeffs = np.random.uniform(-1.0, 1.0, (num_gaussians, num_sh_coeffs, 3)).astype(np.float32)
        colors = None
    else:
        colors = np.random.uniform(0.2, 1.0, (num_gaussians, 3)).astype(np.float32)
        sh_coeffs = None
    
    # Random scales (size of Gaussians)
    scales = np.random.uniform(0.05, 0.2, (num_gaussians, 3)).astype(np.float32)
    
    # Random rotations as quaternions (w, x, y, z)
    rotations = np.zeros((num_gaussians, 4), dtype=np.float32)
    # Identity quaternions for simplicity
    rotations[:, 0] = 1.0  # w component
    
    # Random opacities
    opacities = np.random.uniform(0.5, 1.0, num_gaussians).astype(np.float32)
    
    return means, colors, scales, rotations, opacities, sh_coeffs

def preprocess_gaussians(means, colors, scales, rotations, opacities, sh_coeffs, 
                         view_matrix_np, proj_matrix_np, camera_pos, 
                         focal_x, focal_y, tan_fovx, tan_fovy, 
                         image_width, image_height, sh_degree=3):
    """Preprocess Gaussians for rendering (similar to 3dgs.py)."""
    num_gaussians = means.shape[0]
    
    # Filtered lists for valid Gaussians
    filtered_means = []
    filtered_colors = []
    filtered_scales = []
    filtered_rotations = []
    filtered_opacities = []
    filtered_depths = []
    
    for i in range(num_gaussians):
        # Frustum culling
        p_view = in_frustum(means[i], view_matrix_np)
        if p_view is None:
            continue
        
        # Store depth for sorting
        depth = p_view[2]
        
        # If using spherical harmonics, compute colors
        if sh_coeffs is not None:
            computed_color = compute_sh_color(sh_degree, means[i], camera_pos, sh_coeffs[i])
            current_color = computed_color
        else:
            current_color = colors[i]
        
        # Add to filtered lists
        filtered_means.append(means[i])
        filtered_colors.append(current_color)
        filtered_scales.append(scales[i])
        filtered_rotations.append(rotations[i])
        filtered_opacities.append(opacities[i])
        filtered_depths.append(depth)
    
    # Convert filtered lists to arrays
    if filtered_means:
        filtered_means = np.array(filtered_means, dtype=np.float32)
        filtered_colors = np.array(filtered_colors, dtype=np.float32)
        filtered_scales = np.array(filtered_scales, dtype=np.float32)
        filtered_rotations = np.array(filtered_rotations, dtype=np.float32)
        filtered_opacities = np.array(filtered_opacities, dtype=np.float32)
        filtered_depths = np.array(filtered_depths, dtype=np.float32)
        
        # Sort by depth (back to front for alpha blending)
        depth_indices = np.argsort(filtered_depths)[::-1]  # Descending order
        
        filtered_means = filtered_means[depth_indices]
        filtered_colors = filtered_colors[depth_indices]
        filtered_scales = filtered_scales[depth_indices]
        filtered_rotations = filtered_rotations[depth_indices]
        filtered_opacities = filtered_opacities[depth_indices]
    else:
        # Return empty arrays if no Gaussians passed frustum culling
        filtered_means = np.zeros((0, 3), dtype=np.float32)
        filtered_colors = np.zeros((0, 3), dtype=np.float32)
        filtered_scales = np.zeros((0, 3), dtype=np.float32)
        filtered_rotations = np.zeros((0, 4), dtype=np.float32)
        filtered_opacities = np.zeros(0, dtype=np.float32)
    
    return filtered_means, filtered_colors, filtered_scales, filtered_rotations, filtered_opacities

def render_image(output_path="gaussian_render.png", use_sh=False, sh_degree=3):
    """Render an image of 3D Gaussians."""
    # Camera parameters
    image_width = 800
    image_height = 600
    aspect_ratio = image_width / image_height
    fov_degrees = 45.0
    
    # Camera position and orientation
    camera_pos = [4.0, 3.0, 4.0]
    look_at = [0.0, 0.0, 0.0]
    up = [0.0, 1.0, 0.0]
    
    # Create view and projection matrices
    view_matrix, proj_matrix, view_matrix_np, proj_matrix_np = create_camera_matrices(
        camera_pos, look_at, up, fov_degrees, aspect_ratio
    )
    
    # Calculate tan(fov/2) for x and y directions
    tan_fovx = np.tan(np.radians(fov_degrees / 2)) * aspect_ratio
    tan_fovy = np.tan(np.radians(fov_degrees / 2))
    
    # Calculate focal lengths
    focal_x = image_width / (2 * tan_fovx)
    focal_y = image_height / (2 * tan_fovy)
    
    # Create sample Gaussians
    means, colors, scales, rotations, opacities, sh_coeffs = create_sample_gaussians(
        100, with_sh=use_sh, sh_degree=sh_degree
    )
    
    # Preprocess Gaussians (frustum culling, depth sorting, SH calculation)
    filtered_means, filtered_colors, filtered_scales, filtered_rotations, filtered_opacities = preprocess_gaussians(
        means, colors, scales, rotations, opacities, sh_coeffs,
        view_matrix_np, proj_matrix_np, camera_pos,
        focal_x, focal_y, tan_fovx, tan_fovy,
        image_width, image_height, sh_degree
    )
    
    # Background color
    background = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    
    # Scale modifier
    scale_modifier = 1.0
    
    # Print all parameters for debugging and comparison with 3dgs.py
    print("\n----- DEBUGGING PARAMETERS -----")
    print(f"Image dimensions: {image_width}x{image_height}")
    print(f"Background color: {background}")
    print(f"Number of Gaussians after filtering: {len(filtered_means)}")
    print(f"Gaussian means shape: {filtered_means.shape}")
    print(f"Gaussian colors shape: {filtered_colors.shape}")
    print(f"Gaussian scales shape: {filtered_scales.shape}")
    print(f"Gaussian rotations shape: {filtered_rotations.shape}")
    print(f"Gaussian opacities shape: {filtered_opacities.shape}")
    print(f"Scale modifier: {scale_modifier}")
    print(f"Tan fovx: {tan_fovx}, Tan fovy: {tan_fovy}")
    print(f"Focal length x: {focal_x}, Focal length y: {focal_y}")
    print(f"Camera position: {camera_pos}")
    
    # Print some sample values for comparison
    if len(filtered_means) > 0:
        print("\nSample values for first Gaussian:")
        print(f"  Position: {filtered_means[0]}")
        print(f"  Color: {filtered_colors[0]}")
        print(f"  Scale: {filtered_scales[0]}")
        print(f"  Rotation: {filtered_rotations[0]}")
        print(f"  Opacity: {filtered_opacities[0]}")
    
    # Print view and projection matrices
    print("\nView matrix (numpy):")
    print(view_matrix_np)
    print("\nProjection matrix (numpy):")
    print(proj_matrix_np)
    print("----- END DEBUGGING PARAMETERS -----\n")
    
    exit()
    
    # Call the Gaussian rasterizer
    rendered_image, depth_image = rasterize_gaussians(
        background=background,
        means3d=filtered_means,
        colors=filtered_colors,
        opacities=filtered_opacities,
        scales=filtered_scales,
        rotations=filtered_rotations,
        scale_modifier=scale_modifier,
        view_matrix=view_matrix,
        proj_matrix=proj_matrix,
        tan_fovx=tan_fovx,
        tan_fovy=tan_fovy,
        image_height=image_height,
        image_width=image_width,
        camera_pos=wp.vec3(camera_pos[0], camera_pos[1], camera_pos[2]),
        antialiasing=True
    )
    
    # Convert the rendered image from device to host
    rendered_array = wp.to_torch(rendered_image).cpu().numpy()
    
    # Filter out low values and ensure colors are in valid range
    rendered_array = np.where(rendered_array < 0.1, 0.0, rendered_array)
    rendered_array = np.clip(rendered_array, 0.0, 1.0)
    
    # Save the image
    plt.figure(figsize=(10, 7.5))
    plt.imshow(rendered_array)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Also save as PNG using imageio for better quality
    imageio.imwrite(output_path, (rendered_array * 255).astype(np.uint8))
    
    print(f"Rendered image saved to {output_path}")
    return rendered_array

if __name__ == "__main__":
    # Set up 3 Gaussians exactly like in 3dgs.py
    pts = np.array([[2, 0, -2], [0, 2, -2], [-2, 0, -2]], dtype=np.float32)
    n = len(pts)
    shs = np.random.random((n, 16, 3)).astype(np.float32)
    opacities = np.ones(n, dtype=np.float32)  # Changed from (n,1) to match our format
    scales = np.ones((n, 3), dtype=np.float32)
    rotations = np.zeros((n, 4), dtype=np.float32)  # Using quaternions instead of matrices
    rotations[:, 0] = 1.0  # w component = 1 for identity rotation

    # Set camera parameters exactly like in 3dgs.py
    camera_pos = np.array([0, 0, 5], dtype=np.float32)
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    look_at = np.array([0, 0, 0], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)

    # Image parameters
    image_width = 700
    image_height = 700
    aspect_ratio = image_width / image_height
    fov_degrees = 45.0

    # Create view and projection matrices
    view_matrix, proj_matrix, view_matrix_np, proj_matrix_np = create_camera_matrices(
        camera_pos, look_at, up, fov_degrees, aspect_ratio
    )

    # Calculate tan(fov/2) for x and y directions
    tan_fovx = np.tan(np.radians(fov_degrees / 2))
    tan_fovy = np.tan(np.radians(fov_degrees / 2))

    # Calculate focal lengths
    focal_x = image_width / (2 * tan_fovx)
    focal_y = image_height / (2 * tan_fovy)

    # Preprocess Gaussians
    filtered_means, filtered_colors, filtered_scales, filtered_rotations, filtered_opacities = preprocess_gaussians(
        pts, None, scales, rotations, opacities, shs,
        view_matrix_np, proj_matrix_np, camera_pos,
        focal_x, focal_y, tan_fovx, tan_fovy,
        image_width, image_height, sh_degree=3
    )

    # Background color (black as in 3dgs.py)
    background = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Scale modifier
    scale_modifier = 1.0

    # Print all parameters for debugging and comparison with 3dgs.py
    print("\n----- DEBUGGING PARAMETERS -----")
    print(f"Image dimensions: {image_width}x{image_height}")
    print(f"Background color: {background}")
    print(f"Number of Gaussians after filtering: {len(filtered_means)}")
    print(f"Gaussian means shape: {filtered_means.shape}")
    print(f"Gaussian colors shape: {filtered_colors.shape}")
    print(f"Gaussian scales shape: {filtered_scales.shape}")
    print(f"Gaussian rotations shape: {filtered_rotations.shape}")
    print(f"Gaussian opacities shape: {filtered_opacities.shape}")
    print(f"Scale modifier: {scale_modifier}")
    print(f"Tan fovx: {tan_fovx}, Tan fovy: {tan_fovy}")
    print(f"Focal length x: {focal_x}, Focal length y: {focal_y}")
    print(f"Camera position: {camera_pos}")
    
    # Print some sample values for comparison
    if len(filtered_means) > 0:
        print("\nSample values for first Gaussian:")
        print(f"  Position: {filtered_means[0]}")
        print(f"  Color: {filtered_colors[0]}")
        print(f"  Scale: {filtered_scales[0]}")
        print(f"  Rotation: {filtered_rotations[0]}")
        print(f"  Opacity: {filtered_opacities[0]}")
    
    # Print view and projection matrices
    print("\nView matrix (numpy):")
    print(view_matrix_np)
    print("\nProjection matrix (numpy):")
    print(proj_matrix_np)
    print("----- END DEBUGGING PARAMETERS -----\n")
    
    # Call the Gaussian rasterizer
    rendered_image, depth_image = rasterize_gaussians(
        background=background,
        means3d=filtered_means,
        colors=filtered_colors,
        opacities=filtered_opacities,
        scales=filtered_scales,
        rotations=filtered_rotations,
        scale_modifier=scale_modifier,
        view_matrix=view_matrix,
        proj_matrix=proj_matrix,
        tan_fovx=tan_fovx,
        tan_fovy=tan_fovy,
        image_height=image_height,
        image_width=image_width,
        camera_pos=wp.vec3(camera_pos[0], camera_pos[1], camera_pos[2]),
        antialiasing=True
    )

    # Convert the rendered image from device to host
    rendered_array = wp.to_torch(rendered_image).cpu().numpy()

    # Filter out low values and ensure colors are in valid range
    rendered_array = np.where(rendered_array < 0.1, 0.0, rendered_array)
    rendered_array = np.clip(rendered_array, 0.0, 1.0)

    # Display and save using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(rendered_array)
    plt.axis('off')
    plt.savefig("gaussian_render.png", bbox_inches='tight', dpi=150)
    # plt.show()
    # plt.close()

    # Also save as PNG using imageio for better quality
    imageio.imwrite("gaussian_render.png", (rendered_array * 255).astype(np.uint8))
