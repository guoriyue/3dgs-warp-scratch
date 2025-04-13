import numpy as np
import warp as wp
import matplotlib.pyplot as plt
from wp_kernels import rasterize_gaussians
import imageio
from config import *
import json
import os
import torch
from plyfile import PlyData, PlyElement

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

def load_ply_gaussians(ply_path):
    """Load a PLY file with Gaussian data."""
    print(f"Loading PLY file: {ply_path}")
    
    # Load the PLY file using plyfile
    plydata = PlyData.read(ply_path)
    
    # Extract vertex data
    vertices = plydata['vertex']
    
    # Extract positions
    x = np.array(vertices['x'])
    y = np.array(vertices['y'])
    z = np.array(vertices['z'])
    points = np.column_stack((x, y, z)).astype(np.float32)
    
    # Extract colors if available
    if all(c in vertices.properties for c in ['red', 'green', 'blue']):
        r = np.array(vertices['red']) / 255.0  # Normalize to [0, 1]
        g = np.array(vertices['green']) / 255.0
        b = np.array(vertices['blue']) / 255.0
        colors = np.column_stack((r, g, b)).astype(np.float32)
    else:
        # Default to random colors if not available
        colors = np.random.uniform(0.2, 1.0, (len(points), 3)).astype(np.float32)
    
    # Extract normals if available
    has_normals = all(n in vertices.properties for n in ['nx', 'ny', 'nz'])
    if has_normals:
        nx = np.array(vertices['nx'])
        ny = np.array(vertices['ny'])
        nz = np.array(vertices['nz'])
        normals = np.column_stack((nx, ny, nz)).astype(np.float32)
    else:
        normals = np.zeros_like(points)
    
    # For PLY files without rotation/scale information, we'll use default values
    num_points = points.shape[0]
    
    # Use simple scales (assuming points are already scaled appropriately)
    scales = np.ones((num_points, 3), dtype=np.float32) * 0.01
    
    # Use identity quaternions for rotations
    rotations = np.zeros((num_points, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w component = 1 for identity quaternion
    
    # Use fixed opacity for all points
    opacities = np.ones(num_points, dtype=np.float32) * 0.5
    
    print(f"Loaded {num_points} 3D Gaussians from PLY")
    return points, colors, scales, rotations, opacities

def load_cameras_from_json(json_path):
    """Load camera parameters from a JSON file."""
    with open(json_path, 'r') as f:
        cameras = json.load(f)
    return cameras

def render_ply_with_camera(ply_path, output_path, camera_idx=0, camera_json=None, image_width=800, image_height=600):
    """Render a PLY file from a specific camera position."""
    # Load Gaussian data from PLY
    means, colors, scales, rotations, opacities = load_ply_gaussians(ply_path)
    
    # Camera parameters
    aspect_ratio = image_width / image_height
    fov_degrees = 45.0
    
    # Camera position and orientation
    if camera_json:
        cameras = load_cameras_from_json(camera_json)
        cam = cameras[camera_idx]
        camera_pos = cam["position"]
        # Calculate look_at: position + forward direction
        rot_matrix = np.array(cam["rotation"])
        forward = rot_matrix[2]  # Assuming the third row is the forward direction
        look_at = np.array(camera_pos) + forward
        up = [0.0, 1.0, 0.0]  # Default up direction
        
        # Use camera's actual width/height
        if "width" in cam and "height" in cam:
            image_width = cam["width"]
            image_height = cam["height"]
            aspect_ratio = image_width / image_height
        
        # Use camera's focal length for FOV calculation if available
        if "fx" in cam:
            # Convert focal length to FOV
            fov_degrees = 2 * np.arctan(image_width / (2 * cam["fx"])) * 180 / np.pi
    else:
        # Default camera
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
    
    # Background color
    background = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    
    # Scale modifier
    scale_modifier = 1.0
    
    print(f"Rendering with camera {camera_idx} at position {camera_pos}")
    
    # Call the Gaussian rasterizer
    rendered_image, depth_image = rasterize_gaussians(
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

def create_rotation_animation_from_ply(ply_path, output_path="playroom_animation.mp4", num_frames=60):
    """Create an animation by rotating the camera around the PLY scene."""
    # Camera parameters
    image_width = 800
    image_height = 600
    aspect_ratio = image_width / image_height
    fov_degrees = 45.0
    
    # Load Gaussian data from PLY
    means, colors, scales, rotations, opacities = load_ply_gaussians(ply_path)
    
    # Camera parameters for orbit
    radius = 6.0
    height = 2.0
    
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
    # Path to the PLY file
    ply_path = "playroom/input.ply"
    camera_json = "playroom/cameras.json"
    
    # Render a single image with camera 0
    render_ply_with_camera(
        ply_path=ply_path,
        output_path="playroom_render.png",
        camera_idx=0,
        camera_json=camera_json
    )
    
    # Create an animation by orbiting around the scene
    create_rotation_animation_from_ply(
        ply_path=ply_path,
        output_path="playroom_animation.mp4"
    ) 