import math
import numpy as np
from plyfile import PlyData

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

def load_ply(filename):
    """
    Load a PLY file containing 3D Gaussian point cloud data.
    
    Args:
        filename: Path to the PLY file
        
    Returns:
        Tuple of (points, scales, rotations, opacities, colors, shs)
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
    
    # Extract rotation information if available, otherwise use default
    if all(f"rot_{i}" in verts.dtype.names for i in range(9)):
        rot_matrix_elements = np.stack([verts[f"rot_{i}"] for i in range(9)], axis=-1).astype(np.float32)
        rotations = rot_matrix_elements.reshape(-1, 3, 3)
    else:
        rotations = np.array([np.eye(3)] * points.shape[0], dtype=np.float32)
    
    return points, scales, rotations, opacities, colors, shs

def get_camera_view_proj(R, t, fovx, fovy, znear=0.01, zfar=100.0):
    view_matrix = world_to_view(R, t)
    proj_matrix = projection_matrix(fovx, fovy, znear, zfar)
    return np.dot(proj_matrix, view_matrix), view_matrix

def fx_to_fovx(fx, width):
    return 2 * math.atan(width / (2 * fx))
