import math
import numpy as np
from plyfile import PlyData, PlyElement
import math
import os
import warp as wp


# Function to save point cloud to PLY file
def save_ply(params, filepath, num_points, colors=None):
    # Get numpy arrays
    positions = params['positions'].numpy()
    scales = params['scales'].numpy()
    rotations = params['rotations'].numpy()
    opacities = params['opacities'].numpy()
    shs = params['shs'].numpy()
    
    # Handle colors - either provided or computed from SH coefficients
    if colors is not None:
        # Use provided colors
        if hasattr(colors, 'numpy'):
            colors_np = colors.numpy()
        else:
            colors_np = colors
    else:
        # Compute colors from SH coefficients (DC term only for simplicity)
        # SH DC coefficients are stored in the first coefficient (index 0)
        colors_np = np.zeros((num_points, 3), dtype=np.float32)
        for i in range(num_points):
            # Get DC term from SH coefficients
            sh_dc = shs[i * 16]  # First SH coefficient contains DC term
            # Convert from SH to RGB (simplified - just use DC term)
            colors_np[i] = np.clip(sh_dc + 0.5, 0.0, 1.0)  # Add 0.5 offset and clamp
    
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
        
        # Add RGB colors (convert to 0-255 range)
        color_255 = (
            int(np.clip(colors_np[i][0] * 255, 0, 255)),
            int(np.clip(colors_np[i][1] * 255, 0, 255)),
            int(np.clip(colors_np[i][2] * 255, 0, 255))
        )
        vertex += color_255
        
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
    
    # Add RGB color fields
    vertex_type.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
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
    
