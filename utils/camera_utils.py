import os
import json
import numpy as np

from utils.math_utils import world_to_view, projection_matrix

def load_camera(camera_info):
    
    # Extract camera parameters
    camera_id = camera_info["camera_id"]
    position = np.array(camera_info["position"], dtype=np.float32)
    rotation = np.array(camera_info["rotation"], dtype=np.float32)
    width = camera_info["width"]
    height = camera_info["height"]
    fx = camera_info["fx"]
    fy = camera_info["fy"]
    
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
    full_proj_matrix = view_matrix @ proj_matrix
    # proj_matrix = np.dot(proj_matrix, view_matrix)
    
    # Calculate other parameters
    tan_fovx = np.tan(fovx * 0.5)
    tan_fovy = np.tan(fovy * 0.5)
    
    camera_params = {
        'camera_pos': position,
        'R': R,
        'view_matrix': view_matrix,
        'proj_matrix': proj_matrix,
        'full_proj_matrix': full_proj_matrix,
        'tan_fovx': tan_fovx,
        'tan_fovy': tan_fovy,
        'focal_x': fx,
        'focal_y': fy,
        'width': width,
        'height': height
    }
    
    print(f"Loaded camera {camera_id} from cameras.json")
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
        return load_camera(camera)
    except Exception as e:
        print(f"Error loading camera from cameras.json: {e}")
        return None

