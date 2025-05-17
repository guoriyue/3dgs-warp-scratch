import os
import json
import numpy as np

from utils.math_utils import world_to_view, projection_matrix

# Y down, Z forward
def load_camera(camera_info):
    """Load camera parameters from camera info dictionary"""
    # Extract camera parameters
    camera_id = camera_info["camera_id"]
    camera_to_world = np.asarray(camera_info["camera_to_world"], dtype=np.float64)
    
    # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    camera_to_world[:3, 1:3] *= -1
    
    # Calculate world to camera transform
    world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
    
    
    # Extract rotation and translation
    R = world_to_camera[:3, :3]
    T = world_to_camera[:3, 3]
    
    
    world_to_camera = world_to_camera.T
    
    width = camera_info.get("width")
    height = camera_info.get("height")
    fx = camera_info.get("focal")
    fy = camera_info.get("focal")
    cx = width / 2
    cy = height / 2
    
    # Calculate field of view from focal length
    fovx = 2 * np.arctan(width / (2 * fx))
    fovy = 2 * np.arctan(height / (2 * fy))
    
    # Create view matrix
    view_matrix = world_to_view(R=R, t=T).T
    
    # Create projection matrix
    znear = 0.01
    zfar = 100.0
    proj_matrix = projection_matrix(fovx=fovx, fovy=fovy, znear=znear, zfar=zfar).T
    full_proj_matrix = view_matrix @ proj_matrix
    
    # Calculate other parameters
    tan_fovx = np.tan(fovx * 0.5)
    tan_fovy = np.tan(fovy * 0.5)
    
    camera_center = np.linalg.inv(world_to_camera)[3, :3]
    
    # Handle camera type and distortion
    camera_model = camera_info.get("camera_model", "OPENCV")
    if camera_model == "OPENCV" or camera_model is None:
        camera_type = 0  # PERSPECTIVE
    elif camera_model == "OPENCV_FISHEYE":
        camera_type = 1  # FISHEYE
    else:
        raise ValueError(f"Unsupported camera_model '{camera_model}'")
    
    # Get distortion parameters
    distortion_params = []
    for param_name in ["k1", "k2", "p1", "p2", "k3", "k4"]:
        distortion_params.append(camera_info.get(param_name, 0.0))
    
    camera_params = {
        'R': R,
        'T': T,
        'camera_center': camera_center,
        'view_matrix': view_matrix,
        'proj_matrix': proj_matrix,
        'full_proj_matrix': full_proj_matrix,
        'tan_fovx': tan_fovx,
        'tan_fovy': tan_fovy,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'width': width,
        'height': height,
        'camera_to_world': camera_to_world,
        'world_to_camera': world_to_camera,
        'camera_type': camera_type,
        'distortion_params': np.array(distortion_params, dtype=np.float32)
    }
    
    print(f"Loaded camera {camera_id}")
    return camera_params

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
        
        # Use load_camera to process the camera parameters
        return load_camera(camera)
        
    except Exception as e:
        print(f"Error loading camera from cameras.json: {e}")
        return None

