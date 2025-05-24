import math
import numpy as np
from plyfile import PlyData, PlyElement
import math
import os
import warp as wp


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
    
    


def load_gaussians_from_path(input_path):
    """Load Gaussian data from the specified path"""
    if input_path.endswith('.ply'):
        return load_ply(input_path)
    else:
        raise ValueError(f"Unsupported input path format: {input_path}")

    
# Function to create a new point cloud with a reduced set of points
def compact_point_cloud(params, valid_mask):
    # NOTE: This function performs compaction on the CPU via numpy, which can be inefficient.
    # Copying data between GPU (Warp) and CPU (NumPy) introduces overhead.
    # A more performant approach would implement compaction directly on the GPU,
    # possibly using parallel prefix sums (scan) to determine new indices.
    mask_np = valid_mask.numpy()
    indices = np.where(mask_np)[0]
    num_valid = len(indices)
    
    # Create new arrays
    new_positions = wp.zeros(num_valid, dtype=wp.vec3)
    new_scales = wp.zeros(num_valid, dtype=wp.vec3)
    new_rotations = wp.zeros(num_valid, dtype=wp.vec4)
    new_opacities = wp.zeros(num_valid, dtype=float)
    new_shs = wp.zeros(num_valid * 16, dtype=wp.vec3)
    
    # Copy valid points
    positions_np = params['positions'].numpy()
    scales_np = params['scales'].numpy()
    rotations_np = params['rotations'].numpy()
    opacities_np = params['opacities'].numpy()
    shs_np = params['shs'].numpy()
    
    for i, idx in enumerate(indices):
        new_positions.numpy()[i] = positions_np[idx]
        new_scales.numpy()[i] = scales_np[idx]
        new_rotations.numpy()[i] = rotations_np[idx]
        new_opacities.numpy()[i] = opacities_np[idx]
        
        # Copy SH coefficients
        for j in range(16):
            new_shs.numpy()[i * 16 + j] = shs_np[idx * 16 + j]
    
    # Create new parameters
    new_params = {
        'positions': new_positions,
        'scales': new_scales,
        'rotations': new_rotations,
        'opacities': new_opacities,
        'shs': new_shs
    }
    
    return new_params, num_valid
