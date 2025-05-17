
import math
import numpy as np
import math
import warp as wp


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