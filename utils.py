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
    plydata = PlyData.read(filename)
    verts = plydata['vertex'].data
    points = np.stack([verts['x'], verts['y'], verts['z']], axis=-1).astype(np.float32)
    
    # Assuming SH and other fields are also stored
    colors = np.stack([verts['red'], verts['green'], verts['blue']], axis=-1).astype(np.float32) / 255.0
    opacities = np.array(verts['opacity']).reshape(-1, 1).astype(np.float32) if 'opacity' in verts.dtype.names else np.ones((points.shape[0], 1), dtype=np.float32)
    
    # Optional: Load SH or set random SH
    shs = np.random.random((points.shape[0], 16, 3)).astype(np.float32)

    # Default identity rotation and uniform scale
    rotations = np.array([np.eye(3)] * points.shape[0], dtype=np.float32)
    scales = np.ones((points.shape[0], 3), dtype=np.float32)

    return points, scales, rotations, opacities, colors, shs

def get_camera_view_proj(R, t, fovx, fovy, znear=0.01, zfar=100.0):
    view_matrix = world_to_view(R, t)
    proj_matrix = projection_matrix(fovx, fovy, znear, zfar)
    return np.dot(proj_matrix, view_matrix), view_matrix

def fx_to_fovx(fx, width):
    return 2 * math.atan(width / (2 * fx))
