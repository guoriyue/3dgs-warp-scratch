import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import math
import argparse
import os
import json
from wp_kernels import render_gaussians
from utils import world_to_view, projection_matrix, load_ply
from config import *

# Initialize Warp
wp.init()

def load_gaussians_from_path(input_path):
    """Load Gaussian data from the specified path"""
    if input_path.endswith('.ply'):
        return load_ply(input_path)
    else:
        raise ValueError(f"Unsupported input path format: {input_path}")

def setup_example_camera(image_width=700, image_height=700, fovx=45.0, fovy=45.0, znear=0.01, zfar=100.0):
    """Setup default camera parameters"""
    # Camera position and orientation
    camera_pos = np.array([0, 0, 5], dtype=np.float32)
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    
    # Compute matrices
    view_matrix = world_to_view(R=R, t=camera_pos)
    proj_matrix = projection_matrix(fovx=fovx, fovy=fovy, znear=znear, zfar=zfar)
    proj_matrix = np.dot(proj_matrix, view_matrix)
    
    # Compute FOV parameters
    tan_fovx = math.tan(fovx * 0.5)
    tan_fovy = math.tan(fovy * 0.5)
    
    focal_x = image_width / (2 * tan_fovx)
    focal_y = image_height / (2 * tan_fovy)
    
    camera_params = {
        'camera_pos': camera_pos,
        'R': R,
        'view_matrix': view_matrix,
        'proj_matrix': proj_matrix,
        'tan_fovx': tan_fovx,
        'tan_fovy': tan_fovy,
        'focal_x': focal_x,
        'focal_y': focal_y
    }
    
    return camera_params

def example_gaussians(image_width=700, image_height=700, fovx=45.0, fovy=45.0, znear=0.01, zfar=100.0):
    """Create example Gaussians for testing and debugging"""
    pts = np.array([[2, 0, -2], [0, 2, -2], [-2, 0, -2]], dtype=np.float32)
    n = len(pts)
    
    # Hard-coded SHs for debugging
    shs = np.array([[0.71734341, 0.91905449, 0.49961076],
                [0.08068483, 0.82132256, 0.01301602],
                [0.8335743,  0.31798138, 0.19709007],
                [0.82589597, 0.28206231, 0.790489  ],
                [0.24008527, 0.21312673, 0.53132892],
                [0.19493135, 0.37989934, 0.61886235],
                [0.98106522, 0.28960672, 0.57313965],
                [0.92623716, 0.46034381, 0.5485369 ],
                [0.81660616, 0.7801104,  0.27813915],
                [0.96114063, 0.69872817, 0.68313804],
                [0.95464185, 0.21984855, 0.92912192],
                [0.23503135, 0.29786121, 0.24999751],
                [0.29844887, 0.6327788,  0.05423596],
                [0.08934335, 0.11851827, 0.04186001],
                [0.59331831, 0.919777,   0.71364335],
                [0.83377388, 0.40242542, 0.8792624 ]]*n).reshape(n, 16, 3)
    
    opacities = np.ones((n, 1), dtype=np.float32)
    scales = np.ones((n, 3), dtype=np.float32)
    rotations = np.array([np.eye(3)] * n, dtype=np.float32)
    
    # Reuse the camera setup function
    camera_params = setup_example_camera(
        image_width=image_width,
        image_height=image_height,
        fovx=fovx,
        fovy=fovy,
        znear=znear,
        zfar=zfar
    )
    
    return pts, shs, scales, rotations, opacities, camera_params

def load_camera_from_json(input_path, camera_id=0):
    """Load camera parameters from camera.json file"""
    camera_file = os.path.join(os.path.dirname(input_path), "cameras.json")
    if not os.path.exists(camera_file):
        print(f"Warning: No cameras.json found in {os.path.dirname(input_path)}, using default camera")
        return None
    
    try:
        with open(camera_file, 'r') as f:
            cameras = json.load(f)
        
        # Find camera with specified ID, or use the first one
        camera = next((cam for cam in cameras if cam["id"] == camera_id), cameras[0])
        
        # Extract camera parameters
        position = np.array(camera["position"], dtype=np.float32)
        rotation = np.array(camera["rotation"], dtype=np.float32)
        width = camera["width"]
        height = camera["height"]
        fx = camera["fx"]
        fy = camera["fy"]
        
        # Calculate field of view from focal length
        fovx = 2 * np.arctan(width / (2 * fx))
        fovy = 2 * np.arctan(height / (2 * fy))
        
        # Convert rotation matrix format if needed
        R = rotation  # Adjust if format differs
        
        # Create view matrix
        view_matrix = world_to_view(R=R, t=position)
        
        # Create projection matrix
        znear = 0.01
        zfar = 100.0
        proj_matrix = projection_matrix(fovx=fovx, fovy=fovy, znear=znear, zfar=zfar)
        proj_matrix = np.dot(proj_matrix, view_matrix)
        
        # Calculate other parameters
        tan_fovx = np.tan(fovx * 0.5)
        tan_fovy = np.tan(fovy * 0.5)
        
        camera_params = {
            'camera_pos': position,
            'R': R,
            'view_matrix': view_matrix,
            'proj_matrix': proj_matrix,
            'tan_fovx': tan_fovx,
            'tan_fovy': tan_fovy,
            'focal_x': fx,
            'focal_y': fy,
            'width': width,
            'height': height
        }
        
        print(f"Loaded camera {camera_id} from cameras.json")
        return camera_params
        
    except Exception as e:
        print(f"Error loading camera from cameras.json: {e}")
        return None

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Render 3D Gaussians")
    parser.add_argument("--input_path", type=str, default=None, 
                        help="Path to input data (PLY file or directory)")
    parser.add_argument("--output", type=str, default="gaussian_render.png",
                        help="Output image filename")
    parser.add_argument("--width", type=int, default=700, help="Image width")
    parser.add_argument("--height", type=int, default=700, help="Image height")
    args = parser.parse_args()
    
    # Set image parameters
    image_width = args.width
    image_height = args.height
    fovx = 45.0
    fovy = 45.0
    znear = 0.01
    zfar = 100.0
    background = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale_modifier = 1.0
    
    if args.input_path:
        # Load Gaussians from provided path
        try:
            # pts, scales, rotations, opacities, colors, shs = load_gaussians_from_path(f"{args.input_path}/input.ply")
            pts, scales, rotations, opacities, colors, shs = load_gaussians_from_path(f"/Users/guomingfei/Desktop/warp-nerf-scratch/playroom/point_cloud/iteration_30000/point_cloud.ply")
            
            n = len(pts)
            print(f"Loaded {n} Gaussians from {args.input_path}")
            
            # Try to load camera from cameras.json
            camera_params = load_camera_from_json(f"{args.input_path}/cameras.json", camera_id=1)
   
            # If no camera found, use default camera
            if camera_params is None:
                camera_params = setup_example_camera(
                    image_width=image_width, 
                    image_height=image_height,
                    fovx=fovx,
                    fovy=fovy,
                    znear=znear,
                    zfar=zfar
                )
            else:
                # Update image dimensions from camera if available
                if 'width' in camera_params and 'height' in camera_params:
                    image_width = camera_params['width']
                    image_height = camera_params['height']
                    print(f"Using image dimensions from camera: {image_width}x{image_height}")
            
        except Exception as e:
            print(f"Error loading from path: {e}")
            exit(1)
    else:
        # Use example Gaussians
        pts, shs, scales, rotations, opacities, camera_params = example_gaussians(
            image_width=image_width,
            image_height=image_height,
            fovx=fovx,
            fovy=fovy,
            znear=znear,
            zfar=zfar
        )
        n = len(pts)
        print(f"Using {n} example Gaussians")
    
    # Generate random colors if not loaded from input
    if args.input_path and 'colors' in locals():
        # Use loaded colors if available
        pass
    else:
        colors = np.random.random((n, 3)).astype(np.float32)

    # Call the Gaussian rasterizer
    rendered_image, depth_image = render_gaussians(
        background=background,
        means3D=pts,
        colors=colors,
        opacity=opacities,
        scales=scales,
        rotations=rotations,
        scale_modifier=scale_modifier,
        viewmatrix=camera_params['view_matrix'],
        projmatrix=camera_params['proj_matrix'],
        tan_fovx=camera_params['tan_fovx'],
        tan_fovy=camera_params['tan_fovy'],
        image_height=image_height,
        image_width=image_width,
        sh=shs,
        degree=3,
        campos=camera_params['camera_pos'],
        prefiltered=False,
        antialiasing=False,
        clamped=True,
        debug=True
    )

    # Convert the rendered image from device to host
    rendered_array = wp.to_torch(rendered_image).cpu().numpy()
    
    # Display and save using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(rendered_array)
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    print(f"Rendered image saved to {args.output}")

