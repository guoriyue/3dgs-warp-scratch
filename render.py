import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import math
import argparse
import os
import json
from forward import render_gaussians
from utils import world_to_view, projection_matrix, load_ply, matrix_to_quaternion
from config import DEVICE, RenderParams

# Initialize Warp
wp.init()

def setup_example_camera(image_width=None, image_height=None, fovx=None, fovy=None, znear=None, zfar=None):
    """Setup default camera parameters"""
    # Use default values from RenderParams if not provided
    image_width = image_width or RenderParams.default_width
    image_height = image_height or RenderParams.default_height
    fovx = fovx or RenderParams.default_fovx
    fovy = fovy or RenderParams.default_fovy
    znear = znear or RenderParams.default_znear
    zfar = zfar or RenderParams.default_zfar
    
    # Camera position and orientation from RenderParams
    camera_pos = RenderParams.default_camera_pos
    R = RenderParams.default_camera_R
    
    # Compute matrices
    view_matrix = world_to_view(R=R, t=camera_pos)
    proj_matrix = projection_matrix(fovx=fovx, fovy=fovy, znear=znear, zfar=zfar)
    
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
        'focal_y': focal_y,
        'width': image_width,
        'height': image_height
    }
    
    return camera_params

def example_gaussians(image_width=None, image_height=None, fovx=None, fovy=None, znear=None, zfar=None):
    """Create example Gaussians for testing and debugging"""
    # Use default values from RenderParams if not provided
    image_width = image_width or RenderParams.default_width
    image_height = image_height or RenderParams.default_height
    fovx = fovx or RenderParams.default_fovx
    fovy = fovy or RenderParams.default_fovy
    znear = znear or RenderParams.default_znear
    zfar = zfar or RenderParams.default_zfar
    
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
    
    # Create quaternion rotations (identity quaternions)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 3] = 1.0  # Set w component to 1.0
    
    colors = np.ones((n, 3), dtype=np.float32)
    
    # Reuse the camera setup function
    camera_params = setup_example_camera(
        image_width=image_width,
        image_height=image_height,
        fovx=fovx,
        fovy=fovy,
        znear=znear,
        zfar=zfar
    )
    return pts, shs, scales, colors, rotations, opacities, camera_params



def load_gaussians_from_path(input_path):
    """Load Gaussian data from the specified path"""
    if input_path.endswith('.ply'):
        return load_ply(input_path)
    else:
        raise ValueError(f"Unsupported input path format: {input_path}")


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
        # proj_matrix = np.dot(proj_matrix, view_matrix)
        
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

class GaussianRenderer:
    def __init__(self, params, cameras, config):
        """Initialize the renderer with point cloud parameters and camera data.
        
        Args:
            params: Dictionary containing point cloud parameters (positions, scales, rotations, etc.)
            cameras: List of camera dictionaries containing view/projection matrices and parameters
            config: Dictionary containing rendering configuration
        """
        self.params = params
        self.cameras = cameras
        self.config = config
        
    def render_view(self, camera_idx):
        """Render a view from a specific camera using the current point cloud."""
        camera = self.cameras[camera_idx]
        
        # Get point cloud data as numpy arrays
        positions_np = self.params['positions'].numpy()
        scales_np = self.params['scales'].numpy()
        rotations_np = self.params['rotations'].numpy()
        opacities_np = self.params['opacities'].numpy()
        shs_np = self.params['shs'].numpy()
        
        # Flip z-axis direction for positions
        positions_np = positions_np.copy()  # Create a copy to avoid modifying the original
        positions_np[:, 2] = -positions_np[:, 2]  # Flip z coordinate

        # Render using the warp renderer
        return render_gaussians(
            background=np.array(self.config['background_color'], dtype=np.float32),
            means3D=positions_np,
            colors=None,  # Use SH coefficients instead
            opacity=opacities_np,
            scales=scales_np,
            rotations=rotations_np,
            scale_modifier=self.config['scale_modifier'],
            viewmatrix=camera['view_matrix'],
            projmatrix=camera['proj_matrix'],
            tan_fovx=camera['tan_fovx'],
            tan_fovy=camera['tan_fovy'],
            image_height=camera['height'],
            image_width=camera['width'],
            sh=shs_np,  # Pass SH coefficients
            degree=self.config['sh_degree'],
            campos=camera['camera_pos'],
            prefiltered=False,
            antialiasing=True,
            clamped=True
        )
        
    def render_all_views(self):
        """Render all camera views and return the results."""
        results = []
        for i in range(len(self.cameras)):
            rendered_image, depth_image, _ = self.render_view(i)
            results.append((rendered_image, depth_image))
        return results

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Render 3D Gaussians")
    parser.add_argument("--input_path", type=str, default=None, 
                        help="Path to input data (PLY file or directory)")
    parser.add_argument("--output", type=str, default="gaussian_render.png",
                        help="Output image filename")
    parser.add_argument("--width", type=int, default=RenderParams.default_width, help="Image width")
    parser.add_argument("--height", type=int, default=RenderParams.default_height, help="Image height")
    parser.add_argument("--debug", action="store_true", help="Enable additional debug output")
    args = parser.parse_args()
    
    # Set image parameters
    image_width = args.width
    image_height = args.height
    fovx = RenderParams.default_fovx
    fovy = RenderParams.default_fovy
    znear = RenderParams.default_znear
    zfar = RenderParams.default_zfar
    background = RenderParams.background_color
    scale_modifier = RenderParams.scale_modifier
    
    if args.input_path:
        # Load Gaussians from provided path
        try:
            # pts, scales, rotations, opacities, colors, shs = load_gaussians_from_path(f"{args.input_path}/input.ply")
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
                    
                # Print camera info for debugging
                print(f"Camera parameters:")
                print(f"  - Position: {camera_params['camera_pos']}")
                print(f"  - View matrix: \n{camera_params['view_matrix']}")
                print(f"  - Projection matrix: \n{camera_params['proj_matrix']}")
            
        except Exception as e:
            print(f"Error loading from path: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    else:
        # Use example Gaussians
        pts, shs, scales, colors, rotations, opacities, camera_params = example_gaussians(
            image_width=image_width,
            image_height=image_height,
            fovx=fovx,
            fovy=fovy,
            znear=znear,
            zfar=zfar
        )
        n = len(pts)
        print(f"Using {n} example Gaussians")

    # Debugging: Force a fixed camera for consistency
    if args.debug and args.input_path:
        print("Using a fixed example camera for debugging")
        camera_params = setup_example_camera(
            image_width=image_width, 
            image_height=image_height,
            fovx=fovx,
            fovy=fovy,
            znear=znear,
            zfar=zfar
        )

    print(f"Starting rendering with {n} points to {image_width}x{image_height} image")
    
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
        projmatrix=camera_params['proj_matrix'],
        tan_fovx=camera_params['tan_fovx'],
        tan_fovy=camera_params['tan_fovy'],
        image_height=image_height,
        image_width=image_width,
        sh=shs,
        degree=RenderParams.sh_degree,
        campos=camera_params['camera_pos'],
        prefiltered=RenderParams.prefiltered,
        antialiasing=RenderParams.antialiasing,
        clamped=RenderParams.clamped,
        debug=args.debug or RenderParams.debug
    )

    print("Rendering completed")
    
    # Convert the rendered image from device to host
    rendered_array = wp.to_torch(rendered_image).cpu().numpy()
    
    # Check if the image has any non-background pixels
    bg_color = background
    non_bg_pixels = np.sum(np.any(np.abs(rendered_array - bg_color) > 0.01, axis=2))
    total_pixels = image_width * image_height
    print(f"Image statistics: {non_bg_pixels} / {total_pixels} non-background pixels ({non_bg_pixels/total_pixels*100:.2f}%)")
    
    # Display and save using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(rendered_array)
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    print(f"Rendered image saved to {args.output}")

