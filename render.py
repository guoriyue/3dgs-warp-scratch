import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import math
import argparse
import os
import json
from forward import render_gaussians
from utils.math_utils import world_to_view, projection_matrix, matrix_to_quaternion
from utils.point_cloud_utils import load_ply, load_gaussians_from_path
from utils.camera_utils import load_camera_from_json
from config import DEVICE

# Initialize Warp
wp.init()

def setup_example_scene(image_width=1800, image_height=1800, fovx=45.0, fovy=45.0, znear=0.01, zfar=100.0):
    """Setup example scene with camera and Gaussians for testing and debugging"""
    # Camera setup
    T = np.array([0, 0, 5], dtype=np.float32)
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = R
    world_to_camera[:3, 3] = T
    world_to_camera = world_to_camera.T
    
    # Compute matrices
    view_matrix = world_to_view(R=R, t=T)
    proj_matrix = projection_matrix(fovx=fovx, fovy=fovy, znear=znear, zfar=zfar).T
    full_proj_matrix = world_to_camera @ proj_matrix
    
    camera_center = np.linalg.inv(world_to_camera)[3, :3]
    
    # Compute FOV parameters
    tan_fovx = math.tan(fovx * 0.5)
    tan_fovy = math.tan(fovy * 0.5)
    
    focal_x = image_width / (2 * tan_fovx)
    focal_y = image_height / (2 * tan_fovy)
    
    camera_params = {
        'R': R,
        'T': T,
        'camera_center': camera_center,
        'view_matrix': view_matrix,
        'proj_matrix': proj_matrix,
        'world_to_camera': world_to_camera,
        'full_proj_matrix': full_proj_matrix,
        'tan_fovx': tan_fovx,
        'tan_fovy': tan_fovy,
        'focal_x': focal_x,
        'focal_y': focal_y,
        'width': image_width,
        'height': image_height
    }
    
    # Gaussian setup
    pts = np.array([[-1, -1, -2], [0, 1, -2], [1, -1, -2]], dtype=np.float32)
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
    
    # Create quaternion rotations (identity quaternions)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 3] = 1.0  # Set w component to 1.0
    
    colors = np.ones((n, 3), dtype=np.float32)
    
    return pts, shs, scales, colors, rotations, opacities, camera_params

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Render 3D Gaussians")
    parser.add_argument("--input_path", type=str, default=None, 
                        help="Path to input data (PLY file or directory)")
    parser.add_argument("--output", type=str, default="gaussian_render.png",
                        help="Output image filename")
    parser.add_argument("--width", type=int, default=1800, help="Image width")
    parser.add_argument("--height", type=int, default=1800, help="Image height")
    parser.add_argument("--debug", action="store_true", help="Enable additional debug output")
    args = parser.parse_args()
    
    # Default rendering parameters
    background = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Black background
    scale_modifier = 1.0
    sh_degree = 3
    prefiltered = False
    antialiasing = False
    clamped = True
    
    if args.input_path:
        # Load Gaussians from provided path
        try:
            ply_path = "/Users/guomingfei/Desktop/warp-nerf-scratch/playroom/point_cloud/iteration_30000/point_cloud.ply"
            print(f"Loading point cloud from: {ply_path}")
            pts, scales, rotations, opacities, colors, shs = load_gaussians_from_path(ply_path)
            
            n = len(pts)
            print(f"Loaded {n} points from PLY file")
            print(f"Point cloud statistics:")
            print(f"  - Position range: Min {pts.min(axis=0)}, Max {pts.max(axis=0)}")
            print(f"  - Scale range: Min {scales.min(axis=0)}, Max {scales.max(axis=0)}")
            print(f"  - Opacity range: Min {opacities.min()}, Max {opacities.max()}")
            print(f"  - Quaternion rotation shape: {rotations.shape}")
            
            if args.debug and n > 0:
                # Print out a sample point for debugging
                idx = 0
                print(f"Sample point {idx}:")
                print(f"  - Position: {pts[idx]}")
                print(f"  - Scale: {scales[idx]}")
                print(f"  - Rotation (quaternion x,y,z,w): {rotations[idx]}")
                print(f"  - Opacity: {opacities[idx]}")
                if shs is not None:
                    print(f"  - SH (first coefficient): {shs[idx][0]}")
            
            # Try to load camera from cameras.json
            camera_params = load_camera_from_json(f"{args.input_path}/cameras.json", camera_id=0)
   
            # If no camera found, use default camera
            if camera_params is None:
                print("Using default camera parameters")
                camera_params = setup_example_scene(
                    image_width=args.width, 
                    image_height=args.height
                )[-1]  # Get only the camera_params from the tuple
            else:
                # Update image dimensions from camera if available
                if 'width' in camera_params and 'height' in camera_params:
                    args.width = camera_params['width']
                    args.height = camera_params['height']
                    print(f"Using image dimensions from camera: {args.width}x{args.height}")
                    
                # Print camera info for debugging
                print(f"Camera parameters:")
                print(f"  - Position: {camera_params['camera_center']}")
                print(f"  - View matrix: \n{camera_params['view_matrix']}")
                print(f"  - Projection matrix: \n{camera_params['proj_matrix']}")
            
        except Exception as e:
            print(f"Error loading from path: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    else:
        # Use example scene
        pts, shs, scales, colors, rotations, opacities, camera_params = setup_example_scene(
            image_width=args.width,
            image_height=args.height
        )
        n = len(pts)
        print(f"Using {n} example Gaussians")

    # Debugging: Force a fixed camera for consistency
    if args.debug and args.input_path:
        print("Using a fixed example camera for debugging")
        camera_params = setup_example_scene(
            image_width=args.width, 
            image_height=args.height
        )[-1]  # Get only the camera_params from the tuple

    print(f"Starting rendering with {n} points to {args.width}x{args.height} image")
    
    if args.debug:
        print("Rendering parameters:")
        print("background", background)
        print("pts", pts.shape)
        print("colors", colors.shape)
        print("opacities", opacities.shape)
        print("scales", scales.shape)
        print("rotations", rotations.shape)
        print("scale_modifier", scale_modifier)
        print("viewmatrix", camera_params['view_matrix'])
        print("projmatrix", camera_params['proj_matrix'])
        print("tan_fovx", camera_params['tan_fovx'])
        print("tan_fovy", camera_params['tan_fovy'])
        print("image_height", args.height)
        print("image_width", args.width)
        print("shs", shs.shape)
        print("degree", sh_degree)
        print("campos", camera_params['camera_center'])
        print("prefiltered", prefiltered)
        print("antialiasing", antialiasing)
        print("clamped", clamped)
    
    # Call the Gaussian rasterizer
    rendered_image, depth_image, _ = render_gaussians(
        background=background,
        means3D=pts,
        colors=colors,
        opacity=opacities,
        scales=scales,
        rotations=rotations,
        scale_modifier=scale_modifier,
        viewmatrix=camera_params['view_matrix'],
        projmatrix=camera_params['full_proj_matrix'],
        tan_fovx=camera_params['tan_fovx'],
        tan_fovy=camera_params['tan_fovy'],
        image_height=args.height,
        image_width=args.width,
        sh=shs,
        degree=sh_degree,
        campos=camera_params['camera_center'],
        prefiltered=prefiltered,
        antialiasing=antialiasing,
        clamped=clamped,
        debug=args.debug
    )

    print("Rendering completed")
    
    # Convert the rendered image from device to host
    rendered_array = wp.to_torch(rendered_image).cpu().numpy()
    
    # Check if the image has any non-background pixels
    bg_color = background
    non_bg_pixels = np.sum(np.any(np.abs(rendered_array - bg_color) > 0.01, axis=2))
    total_pixels = args.width * args.height
    print(f"Image statistics: {non_bg_pixels} / {total_pixels} non-background pixels ({non_bg_pixels/total_pixels*100:.2f}%)")
    
    # Display and save using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(rendered_array)
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    print(f"Rendered image saved to {args.output}")

