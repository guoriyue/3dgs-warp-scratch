import math
import numpy as np
from plyfile import PlyData, PlyElement
import math

def world_to_view(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
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

def projection_matrix(fovx, fovy, znear, zfar):
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

def matrix_to_quaternion(matrix):
    """
    Convert a 3x3 rotation matrix to a quaternion in (x, y, z, w) format.
    
    Args:
        matrix: 3x3 rotation matrix
        
    Returns:
        Quaternion as (x, y, z, w) in numpy array of shape (4,)
    """
    # Ensure the input is a proper rotation matrix
    # This is just a simple check that might be helpful during debug
    if np.abs(np.linalg.det(matrix) - 1.0) > 1e-5:
        print(f"Warning: Input matrix determinant is not 1: {np.linalg.det(matrix)}")
    
    trace = np.trace(matrix)
    if trace > 0:
        S = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * S
        x = (matrix[2, 1] - matrix[1, 2]) / S
        y = (matrix[0, 2] - matrix[2, 0]) / S
        z = (matrix[1, 0] - matrix[0, 1]) / S
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        S = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / S
        x = 0.25 * S
        y = (matrix[0, 1] + matrix[1, 0]) / S
        z = (matrix[0, 2] + matrix[2, 0]) / S
    elif matrix[1, 1] > matrix[2, 2]:
        S = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / S
        x = (matrix[0, 1] + matrix[1, 0]) / S
        y = 0.25 * S
        z = (matrix[1, 2] + matrix[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / S
        x = (matrix[0, 2] + matrix[2, 0]) / S
        y = (matrix[1, 2] + matrix[2, 1]) / S
        z = 0.25 * S
    
    # Return as (x, y, z, w) to match Warp's convention
    return np.array([x, y, z, w], dtype=np.float32)

def load_ply(filename):
    """
    Load a PLY file containing 3D Gaussian point cloud data.
    
    Args:
        filename: Path to the PLY file
        
    Returns:
        Tuple of (points, scales, rotations, opacities, colors, shs)
        where rotations are quaternions (x, y, z, w)
    """
    plydata = PlyData.read(filename)
    verts = plydata['vertex'].data
    
    # Extract position data
    points = np.stack([verts['x'], verts['y'], verts['z']], axis=-1).astype(np.float32)
    
    # Extract color data if available, otherwise use default
    colors = np.stack([verts['red'], verts['green'], verts['blue']], axis=-1).astype(np.float32) / 255.0 if all(c in verts.dtype.names for c in ['red', 'green', 'blue']) else np.ones((points.shape[0], 3), dtype=np.float32) * 0.5
    
    # Extract opacity if available, otherwise use default
    opacities = np.array(verts['opacity']).reshape(-1, 1).astype(np.float32) if 'opacity' in verts.dtype.names else np.ones((points.shape[0], 1), dtype=np.float32)
    
    # Extract spherical harmonic coefficients if available, otherwise use random values
    shs = np.zeros((points.shape[0], 16, 3), dtype=np.float32)
    if all(name in verts.dtype.names for name in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        shs[:, 0, 0] = verts["f_dc_0"]
        shs[:, 0, 1] = verts["f_dc_1"]
        shs[:, 0, 2] = verts["f_dc_2"]
        rest_fields = [name for name in verts.dtype.names if name.startswith("f_rest_")]
        if rest_fields:
            rest_fields = sorted(rest_fields, key=lambda x: int(x.split('_')[-1]))
            for i, field in enumerate(rest_fields):
                sh_idx = i // 3 + 1
                color_idx = i % 3
                shs[:, sh_idx, color_idx] = verts[field]
    else:
        shs = np.random.random((points.shape[0], 16, 3)).astype(np.float32)
    
    # Extract scale information if available, otherwise use default
    scales = np.stack([verts["scale_0"], verts["scale_1"], verts["scale_2"]], axis=-1).astype(np.float32) if all(f"scale_{i}" in verts.dtype.names for i in range(3)) else np.ones((points.shape[0], 3), dtype=np.float32)
    
    # Check for rotation format:
    # First check if quaternion components exist
    if all(comp in verts.dtype.names for comp in ['rot_x', 'rot_y', 'rot_z', 'rot_w']):
        # Quaternion format (x, y, z, w)
        rotations = np.stack([verts['rot_x'], verts['rot_y'], verts['rot_z'], verts['rot_w']], axis=-1).astype(np.float32)
    else:
        # Default to identity quaternion (0, 0, 0, 1)
        rotations = np.zeros((points.shape[0], 4), dtype=np.float32)
        rotations[:, 3] = 1.0  # Set w component to 1.0
        
    return points, scales, rotations, opacities, colors, shs

def get_camera_view_proj(R, t, fovx, fovy, znear=0.01, zfar=100.0):
    view_matrix = world_to_view(R, t)
    proj_matrix = projection_matrix(fovx, fovy, znear, zfar)
    return np.dot(proj_matrix, view_matrix), view_matrix

def fx_to_fovx(fx, width):
    return 2 * math.atan(width / (2 * fx))



# Function to save point cloud to PLY file
def save_ply(params, filepath, num_points):
    # Get numpy arrays
    positions = params['positions'].numpy()
    scales = params['scales'].numpy()
    rotations = params['rotations'].numpy()
    opacities = params['opacities'].numpy()
    shs = params['shs'].numpy()
    
    # Create vertex data
    vertex_data = []
    for i in range(num_points):
        # Basic properties
        vertex = (
            positions[i][0], positions[i][1], positions[i][2],
            scales[i][0], scales[i][1], scales[i][2],
            opacities[i]
        )
        
        # Add rotation quaternion elements
        quat = rotations[i]
        rot_elements = (quat[0], quat[1], quat[2], quat[3])  # x, y, z, w
        vertex += rot_elements
        
        # Add SH coefficients
        sh_dc = tuple(shs[i * 16][j] for j in range(3))
        vertex += sh_dc
        
        # Add remaining SH coefficients
        sh_rest = []
        for j in range(1, 16):
            for c in range(3):
                sh_rest.append(shs[i * 16 + j][c])
        vertex += tuple(sh_rest)
        
        vertex_data.append(vertex)
    
    # Define the structure of the PLY file
    vertex_type = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('opacity', 'f4')
    ]
    
    # Add rotation quaternion elements
    vertex_type.extend([('rot_x', 'f4'), ('rot_y', 'f4'), ('rot_z', 'f4'), ('rot_w', 'f4')])
    
    # Add SH coefficients
    vertex_type.extend([('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')])
    
    # Add remaining SH coefficients
    for i in range(45):  # 15 coeffs * 3 channels
        vertex_type.append((f'f_rest_{i}', 'f4'))
    
    vertex_array = np.array(vertex_data, dtype=vertex_type)
    el = PlyElement.describe(vertex_array, 'vertex')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the PLY file
    PlyData([el], text=False).write(filepath)
    print(f"Point cloud saved to {filepath}")