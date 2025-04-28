import warp as wp

# Define structures for Gaussian parameters
@wp.struct
class GaussianParams:
    positions: wp.array(dtype=wp.vec3)
    scales: wp.array(dtype=wp.vec3)
    rotations: wp.array(dtype=wp.mat33)
    opacities: wp.array(dtype=float)
    shs: wp.array(dtype=wp.vec3)  # Flattened SH coefficients
