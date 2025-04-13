import numpy as np
import warp as wp
import matplotlib.pyplot as plt
from wp_kernels import rasterize_gaussians
import imageio
from config import *

# Initialize Warp
wp.init()

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
    
    return wp.mat44(view_matrix.flatten()), wp.mat44(projection_matrix.flatten())

def create_sample_gaussians(num_gaussians=50):
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
    
    # Random colors
    colors = np.random.uniform(0.2, 1.0, (num_gaussians, 3)).astype(np.float32)
    
    # Random scales (size of Gaussians)
    scales = np.random.uniform(0.05, 0.2, (num_gaussians, 3)).astype(np.float32)
    
    # Random rotations as quaternions (w, x, y, z)
    rotations = np.zeros((num_gaussians, 4), dtype=np.float32)
    # Identity quaternions for simplicity
    rotations[:, 0] = 1.0  # w component
    
    # Random opacities
    opacities = np.random.uniform(0.5, 1.0, num_gaussians).astype(np.float32)
    
    return means, colors, scales, rotations, opacities

def render_image(output_path="gaussian_render.png"):
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
    view_matrix, proj_matrix = create_camera_matrices(
        camera_pos, look_at, up, fov_degrees, aspect_ratio
    )
    
    # Calculate tan(fov/2) for x and y directions
    tan_fovx = np.tan(np.radians(fov_degrees / 2)) * aspect_ratio
    tan_fovy = np.tan(np.radians(fov_degrees / 2))
    
    # Create sample Gaussians
    means, colors, scales, rotations, opacities = create_sample_gaussians(100)
    
    # Background color
    background = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    
    # Scale modifier
    scale_modifier = 1.0
    
    # Call the Gaussian rasterizer
    rendered_image, depth_image = rasterize_gaussians(
        background=background,
        means3d=means,
        colors=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        scale_modifier=scale_modifier,
        # cov3D_precomp
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
    
    # Ensure colors are in valid range
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

def create_rotation_animation(output_path="gaussian_animation.mp4", num_frames=60):
    """Create an animation by rotating the camera around the scene."""
    # Camera parameters
    image_width = 800
    image_height = 600
    aspect_ratio = image_width / image_height
    fov_degrees = 45.0
    
    # Camera parameters for orbit
    radius = 6.0
    height = 2.0
    
    # Create sample Gaussians (fixed for the animation)
    means, colors, scales, rotations, opacities = create_sample_gaussians(100)
    
    # Background color
    background = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    
    # Scale modifier
    scale_modifier = 1.0
    
    # Calculate tan(fov/2) for x and y directions
    tan_fovx = np.tan(np.radians(fov_degrees / 2)) * aspect_ratio
    tan_fovy = np.tan(np.radians(fov_degrees / 2))
    
    # Create frames
    frames = []
    for i in range(num_frames):
        # Calculate camera position on a circle
        angle = 2 * np.pi * i / num_frames
        camera_pos = [
            radius * np.cos(angle),
            height,
            radius * np.sin(angle)
        ]
        look_at = [0.0, 0.0, 0.0]
        up = [0.0, 1.0, 0.0]
        
        # Create view and projection matrices
        view_matrix, proj_matrix = create_camera_matrices(
            camera_pos, look_at, up, fov_degrees, aspect_ratio
        )
        
        # Call the Gaussian rasterizer
        rendered_image, _ = rasterize_gaussians(
            background=background,
            means3d=means,
            colors=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
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
        
        # Ensure colors are in valid range and convert to uint8
        rendered_array = np.clip(rendered_array, 0.0, 1.0)
        frame = (rendered_array * 255).astype(np.uint8)
        frames.append(frame)
        
        print(f"Rendered frame {i+1}/{num_frames}")
    
    # Save the animation
    imageio.mimsave(output_path, frames, fps=30)
    print(f"Animation saved to {output_path}")

if __name__ == "__main__":
    # Render a single image
    render_image("gaussian_render.png")
    
    # Uncomment to create an animation
    # create_rotation_animation("gaussian_animation.mp4")
