import warp as wp
from config import DEVICE


@wp.func
def wp_vec3_mul_element(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(a[0] * b[0], a[1] * b[1], a[2] * b[2])

# Reinstate the element-wise vector square root helper function
@wp.func
def wp_vec3_sqrt(a: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.sqrt(a[0]), wp.sqrt(a[1]), wp.sqrt(a[2]))

# Add element-wise vector division helper function
@wp.func
def wp_vec3_div_element(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    # Add small epsilon to denominator to prevent division by zero
    # (although Adam's epsilon should mostly handle this)
    safe_b = wp.vec3(b[0] + 1e-9, b[1] + 1e-9, b[2] + 1e-9)
    return wp.vec3(a[0] / safe_b[0], a[1] / safe_b[1], a[2] / safe_b[2])

@wp.func
def wp_vec3_add_element(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2])

@wp.func
def wp_vec3_clamp(x: wp.vec3, min_val: float, max_val: float) -> wp.vec3:
    return wp.vec3(
        wp.clamp(x[0], min_val, max_val),
        wp.clamp(x[1], min_val, max_val),
        wp.clamp(x[2], min_val, max_val)
    )

def to_warp_array(data, dtype, shape_check=None, flatten=False):
    if isinstance(data, wp.array):
        return data
    if data is None:
        return None
    # Convert torch tensor to numpy if needed
    if hasattr(data, 'cpu') and hasattr(data, 'numpy'):
        data = data.cpu().numpy()
    if flatten and data.ndim == 2 and data.shape[1] == 1:
        data = data.flatten()
    return wp.array(data, dtype=dtype, device=DEVICE)


