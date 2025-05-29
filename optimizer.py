import warp as wp
from utils.wp_utils import to_warp_array, wp_vec3_mul_element, wp_vec3_add_element, wp_vec3_sqrt, wp_vec3_div_element, wp_vec3_clamp
from config import *

@wp.kernel
def adam_update(
    # Parameters
    positions: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.vec4),
    opacities: wp.array(dtype=float),
    shs: wp.array(dtype=wp.vec3),
    
    # Gradients
    pos_grads: wp.array(dtype=wp.vec3),
    scale_grads: wp.array(dtype=wp.vec3),
    rot_grads: wp.array(dtype=wp.vec4),
    opacity_grads: wp.array(dtype=float),
    sh_grads: wp.array(dtype=wp.vec3),
    
    # First moments (m)
    m_positions: wp.array(dtype=wp.vec3),
    m_scales: wp.array(dtype=wp.vec3),
    m_rotations: wp.array(dtype=wp.vec4),
    m_opacities: wp.array(dtype=float),
    m_shs: wp.array(dtype=wp.vec3),
    
    # Second moments (v)
    v_positions: wp.array(dtype=wp.vec3),
    v_scales: wp.array(dtype=wp.vec3),
    v_rotations: wp.array(dtype=wp.vec4),
    v_opacities: wp.array(dtype=float),
    v_shs: wp.array(dtype=wp.vec3),
    
    num_points: int,
    lr_pos: float,
    lr_scale: float,
    lr_rot: float,
    lr_opac: float,
    lr_sh: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    iteration: int
):
    i = wp.tid()
    if i >= num_points:
        return
    
    # Bias correction terms
    bias_correction1 = 1.0 - wp.pow(beta1, float(iteration + 1))
    bias_correction2 = 1.0 - wp.pow(beta2, float(iteration + 1))
    
    # Update positions
    m_positions[i] = beta1 * m_positions[i] + (1.0 - beta1) * pos_grads[i]
    # Use the helper function for element-wise multiplication
    v_positions[i] = beta2 * v_positions[i] + (1.0 - beta2) * wp_vec3_mul_element(pos_grads[i], pos_grads[i])
    # Use distinct names for corrected moments per parameter type
    m_pos_corrected = m_positions[i] / bias_correction1
    v_pos_corrected = v_positions[i] / bias_correction2
    # Use the helper function for element-wise sqrt and division
    denominator_pos = wp_vec3_sqrt(v_pos_corrected) + wp.vec3(epsilon, epsilon, epsilon)
    positions[i] = positions[i] - lr_pos * wp_vec3_div_element(m_pos_corrected, denominator_pos)
    
    # Update scales (with some constraints to keep them positive)
    m_scales[i] = beta1 * m_scales[i] + (1.0 - beta1) * scale_grads[i]
    # Use the helper function for element-wise multiplication
    v_scales[i] = beta2 * v_scales[i] + (1.0 - beta2) * wp_vec3_mul_element(scale_grads[i], scale_grads[i])
    # Use distinct names for corrected moments per parameter type
    m_scale_corrected = m_scales[i] / bias_correction1
    v_scale_corrected = v_scales[i] / bias_correction2
    # Use the helper function for element-wise sqrt and division
    denominator_scale = wp_vec3_sqrt(v_scale_corrected) + wp.vec3(epsilon, epsilon, epsilon)
    scale_update = lr_scale * wp_vec3_div_element(m_scale_corrected, denominator_scale)
    scales[i] = wp.vec3(
        wp.max(scales[i][0] - scale_update[0], 0.001),
        wp.max(scales[i][1] - scale_update[1], 0.001),
        wp.max(scales[i][2] - scale_update[2], 0.001)
    )
    
    # Update rotations
    m_rotations[i] = beta1 * m_rotations[i] + (1.0 - beta1) * rot_grads[i]
    # Element-wise multiplication for quaternions
    v_rotations[i] = beta2 * v_rotations[i] + (1.0 - beta2) * wp.vec4(
        rot_grads[i][0] * rot_grads[i][0],
        rot_grads[i][1] * rot_grads[i][1],
        rot_grads[i][2] * rot_grads[i][2],
        rot_grads[i][3] * rot_grads[i][3]
    )
    m_rot_corrected = m_rotations[i] / bias_correction1
    v_rot_corrected = v_rotations[i] / bias_correction2
    # Element-wise sqrt and division for quaternions
    denominator_rot = wp.vec4(
        wp.sqrt(v_rot_corrected[0]) + epsilon,
        wp.sqrt(v_rot_corrected[1]) + epsilon,
        wp.sqrt(v_rot_corrected[2]) + epsilon,
        wp.sqrt(v_rot_corrected[3]) + epsilon
    )
    rot_update = wp.vec4(
        lr_rot * m_rot_corrected[0] / denominator_rot[0],
        lr_rot * m_rot_corrected[1] / denominator_rot[1],
        lr_rot * m_rot_corrected[2] / denominator_rot[2],
        lr_rot * m_rot_corrected[3] / denominator_rot[3]
    )
    rotations[i] = rotations[i] - rot_update
    
    # Normalize quaternion to ensure it's a valid rotation
    quat_length = wp.sqrt(rotations[i][0]*rotations[i][0] + 
                         rotations[i][1]*rotations[i][1] + 
                         rotations[i][2]*rotations[i][2] + 
                         rotations[i][3]*rotations[i][3])
    
    if quat_length > 0.0:
        rotations[i] = wp.vec4(
            rotations[i][0] / quat_length,
            rotations[i][1] / quat_length,
            rotations[i][2] / quat_length,
            rotations[i][3] / quat_length
        )
    
    # Update opacity (with clamping to [0,1])
    m_opacities[i] = beta1 * m_opacities[i] + (1.0 - beta1) * opacity_grads[i]
    # Opacity is scalar, direct multiplication is fine
    v_opacities[i] = beta2 * v_opacities[i] + (1.0 - beta2) * (opacity_grads[i] * opacity_grads[i])
    # Use distinct names for corrected moments per parameter type
    m_opacity_corrected = m_opacities[i] / bias_correction1
    v_opacity_corrected = v_opacities[i] / bias_correction2
    # Opacity is scalar, direct wp.sqrt is fine here
    opacity_update = lr_opac * m_opacity_corrected / (wp.sqrt(v_opacity_corrected) + epsilon)
    opacities[i] = wp.max(wp.min(opacities[i] - opacity_update, 1.0), 0.0)
    
    # Update SH coefficients
    for j in range(16):
        idx = i * 16 + j
        m_shs[idx] = beta1 * m_shs[idx] + (1.0 - beta1) * sh_grads[idx]
        # Use the helper function for element-wise multiplication
        v_shs[idx] = beta2 * v_shs[idx] + (1.0 - beta2) * wp_vec3_mul_element(sh_grads[idx], sh_grads[idx])
        # Use distinct names for corrected moments per parameter type
        m_sh_corrected = m_shs[idx] / bias_correction1
        v_sh_corrected = v_shs[idx] / bias_correction2
        # Use the helper function for element-wise sqrt and division
        denominator_sh = wp_vec3_sqrt(v_sh_corrected) + wp.vec3(epsilon, epsilon, epsilon)
        shs[idx] = shs[idx] - lr_sh * wp_vec3_div_element(m_sh_corrected, denominator_sh)


@wp.kernel
def reset_opacities(
    opacities: wp.array(dtype=float),
    max_opacity: float,
    num_points: int
):
    """Reset opacities to prevent oversaturation."""
    i = wp.tid()
    if i >= num_points:
        return
    
    # Reset opacity to a small value
    opacities[i] = max_opacity

@wp.kernel
def reset_densification_stats(
    xyz_gradient_accum: wp.array(dtype=float),
    denom: wp.array(dtype=float),
    max_radii2D: wp.array(dtype=float),
    num_points: int
):
    """Reset densification statistics after parameter count changes."""
    i = wp.tid()
    if i >= num_points:
        return
    
    xyz_gradient_accum[i] = 0.0
    denom[i] = 0.0
    max_radii2D[i] = 0.0


@wp.kernel
def mark_split_candidates(
    grads: wp.array(dtype=float),
    scales: wp.array(dtype=wp.vec3),
    grad_threshold: float,
    scene_extent: float,
    percent_dense: float,
    split_mask: wp.array(dtype=int),
    num_points: int
):
    """Mark large Gaussians with high gradients for splitting."""
    i = wp.tid()
    if i >= num_points:
        return
    
    # Check if gradient exceeds threshold
    high_grad = grads[i] >= grad_threshold
    
    # Check if Gaussian is large (max scale > threshold)
    max_scale = wp.max(wp.max(scales[i][0], scales[i][1]), scales[i][2])
    scale_threshold = percent_dense * scene_extent
    large_gaussian = max_scale > scale_threshold
    
    # Mark for splitting if both conditions are met
    if (high_grad and large_gaussian):
        split_mask[i] = 1 
    else:
        split_mask[i] = 0

@wp.kernel
def mark_clone_candidates(
    grads: wp.array(dtype=float),
    scales: wp.array(dtype=wp.vec3),
    grad_threshold: float,
    scene_extent: float,
    percent_dense: float,
    clone_mask: wp.array(dtype=int),
    num_points: int
):
    """Mark small Gaussians with high gradients for cloning."""
    i = wp.tid()
    if i >= num_points:
        return
    
    # Check if gradient exceeds threshold
    high_grad = grads[i] >= grad_threshold
    
    # Check if Gaussian is small (max scale <= threshold)
    max_scale = wp.max(wp.max(scales[i][0], scales[i][1]), scales[i][2])
    scale_threshold = percent_dense * scene_extent
    small_gaussian = max_scale <= scale_threshold
    
    # Mark for cloning if both conditions are met
    if (high_grad and small_gaussian):
        clone_mask[i] = 1 
    else:
        clone_mask[i] = 0

@wp.kernel
def split_gaussians(
    split_mask: wp.array(dtype=int),
    prefix_sum: wp.array(dtype=int),
    positions: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.vec4),
    opacities: wp.array(dtype=float),
    shs: wp.array(dtype=wp.vec3),
    N_split: int,
    scale_factor: float,
    offset: int,
    out_positions: wp.array(dtype=wp.vec3),
    out_scales: wp.array(dtype=wp.vec3),
    out_rotations: wp.array(dtype=wp.vec4),
    out_opacities: wp.array(dtype=float),
    out_shs: wp.array(dtype=wp.vec3)
):
    """Split large Gaussians into multiple smaller ones."""
    i = wp.tid()
    
    # Copy original Gaussians first
    if i < len(positions):
        out_positions[i] = positions[i]
        out_scales[i] = scales[i]
        out_rotations[i] = rotations[i]
        out_opacities[i] = opacities[i]
        
        # Copy SH coefficients
        for j in range(16):
            out_shs[i * 16 + j] = shs[i * 16 + j]
    
    # Handle splits
    if i >= len(positions):
        return
        
    if split_mask[i] == 1:
        # Find where to write new Gaussians
        split_idx = prefix_sum[i]
        
        # Create N_split new Gaussians
        for j in range(N_split):
            new_idx = offset + split_idx * N_split + j
            if new_idx < len(out_positions):
                # Scale down the original Gaussian
                scaled_scales = wp.vec3(
                    scales[i][0] * scale_factor,
                    scales[i][1] * scale_factor,
                    scales[i][2] * scale_factor
                )
                
                # Add small random offset for position
                random_offset = wp.vec3(
                    ((wp.randf(wp.uint32(new_idx * 3))) * 2.0 - 1.0) * 0.01,
                    ((wp.randf(wp.uint32(new_idx * 3 + 1))) * 2.0 - 1.0) * 0.01,
                    ((wp.randf(wp.uint32(new_idx * 3 + 2))) * 2.0 - 1.0) * 0.01
                )
                
                out_positions[new_idx] = positions[i] + random_offset
                out_scales[new_idx] = scaled_scales
                out_rotations[new_idx] = rotations[i]
                out_opacities[new_idx] = opacities[i]
                
                # Copy SH coefficients
                for k in range(16):
                    out_shs[new_idx * 16 + k] = shs[i * 16 + k]


@wp.kernel
def clone_gaussians(
    clone_mask: wp.array(dtype=int),
    prefix_sum: wp.array(dtype=int),
    positions: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.vec4),
    opacities: wp.array(dtype=float),
    shs: wp.array(dtype=wp.vec3),  # shape: [N * 16]

    noise_scale: float,
    offset: int,  # where to start writing new points
    out_positions: wp.array(dtype=wp.vec3),
    out_scales: wp.array(dtype=wp.vec3),
    out_rotations: wp.array(dtype=wp.vec4),
    out_opacities: wp.array(dtype=float),
    out_shs: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if i >= offset:
        return

    # Copy original to out[i]
    out_positions[i] = positions[i]
    out_scales[i] = scales[i]
    out_rotations[i] = rotations[i]
    out_opacities[i] = opacities[i]
    for j in range(16):
        out_shs[i * 16 + j] = shs[i * 16 + j]

    if clone_mask[i] == 1:
        base_idx = prefix_sum[i] + offset
        pos = positions[i]
        scale = scales[i]
        rot = rotations[i]
        opac = opacities[i]


        noise = wp.vec3(
            wp.randf(wp.uint32(i * 3)) * noise_scale,
            wp.randf(wp.uint32(i * 3 + 1)) * noise_scale,
            wp.randf(wp.uint32(i * 3 + 2)) * noise_scale
        )

        out_positions[base_idx] = pos + noise
        out_scales[base_idx] = scale
        out_rotations[base_idx] = rot
        out_opacities[base_idx] = opac

        for j in range(16):
            out_shs[base_idx * 16 + j] = shs[i * 16 + j]

@wp.kernel
def prune_gaussians(
    opacities: wp.array(dtype=float),
    opacity_threshold: float,
    valid_mask: wp.array(dtype=int),
    num_points: int
):
    i = wp.tid()
    if i >= num_points:
        return
    # Mark Gaussians for keeping or removal
    if opacities[i] > opacity_threshold:
        valid_mask[i] = 1
    else:
        valid_mask[i] = 0

@wp.kernel
def compact_gaussians(
    valid_mask: wp.array(dtype=int),
    prefix_sum: wp.array(dtype=int),
    positions: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.vec4),
    opacities: wp.array(dtype=float),
    shs: wp.array(dtype=wp.vec3),  # shape: [N * 16]

    out_positions: wp.array(dtype=wp.vec3),
    out_scales: wp.array(dtype=wp.vec3),
    out_rotations: wp.array(dtype=wp.vec4),
    out_opacities: wp.array(dtype=float),
    out_shs: wp.array(dtype=wp.vec3)
):
    i = wp.tid()
    if valid_mask[i] == 0:
        return

    new_i = prefix_sum[i]

    out_positions[new_i] = positions[i]
    out_scales[new_i] = scales[i]
    out_rotations[new_i] = rotations[i]
    out_opacities[new_i] = opacities[i]

    for j in range(16):
        out_shs[new_i * 16 + j] = shs[i * 16 + j]

