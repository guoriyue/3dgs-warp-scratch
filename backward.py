# Add these to wp_kernels.py

import numpy as np
import warp as wp
import math
from config import * # Assuming TILE_M, TILE_N, VEC6, DEVICE are defined here

# Initialize Warp if not already done elsewhere
# wp.init()


# --- Spherical Harmonics Constants ---
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199

@wp.func
def gradient_normalize_wp(v: wp.vec3, grad_out: wp.vec3, epsilon: float=1e-8) -> wp.vec3:
    """ Computes gradient of normalize(v) w.r.t v, scaled by grad_out. """
    v_len = wp.length(v)
    if v_len < epsilon:
        return wp.vec3(0.0, 0.0, 0.0)
    inv_len = 1.0 / v_len
    u = v * inv_len
    u_dot_grad = wp.dot(u, grad_out)
    grad_in = (grad_out - u * u_dot_grad) * inv_len
    return grad_in

@wp.func
def gradient_normalize_quat_wp(v: wp.vec4, grad_out: wp.vec4, epsilon: float=1e-8) -> wp.vec4:
    """ Computes gradient of normalize(v) w.r.t v for vec4, scaled by grad_out. """
    v_len = wp.length(v)
    if v_len < epsilon:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)
    inv_len = 1.0 / v_len
    u = v * inv_len
    u_dot_grad = wp.dot(u, grad_out)
    grad_in = (grad_out - u * u_dot_grad) * inv_len
    return grad_in

@wp.func
def quat_to_mat33_wp(q: wp.vec4) -> wp.mat33:
    """ Converts quaternion (x, y, z, w) to 3x3 rotation matrix (column-major). """
    # Assumes input q is normalized or normalization is handled elsewhere
    x = q[0]; y = q[1]; z = q[2]; r = q[3] # w=r
    x2 = x*x; y2 = y*y; z2 = z*z
    xy = x*y; xz = x*z; yz = y*z
    rx = r*x; ry = r*y; rz = r*z

    # Row-major elements
    r00 = 1.0 - 2.0*(y2 + z2); r01 = 2.0*(xy - rz); r02 = 2.0*(xz + ry)
    r10 = 2.0*(xy + rz); r11 = 1.0 - 2.0*(x2 + z2); r12 = 2.0*(yz - rx)
    r20 = 2.0*(xz - ry); r21 = 2.0*(yz + rx); r22 = 1.0 - 2.0*(x2 + y2)

    # Construct column-major
    return wp.mat33(r00, r10, r20, r01, r11, r21, r02, r12, r22)


# --- Backward Kernels ---
@wp.kernel
def sh_backward_kernel(
    # --- Inputs ---
    num_points: int,
    degree: int, # SH degree used in forward
    means: wp.array(dtype=wp.vec3),      # (N, 3)
    shs: wp.array(dtype=wp.vec3),        # Flattened SH coeffs (N * 16, 3)
    radii: wp.array(dtype=int),             # Radii computed in forward (N,) - used for skipping
    campos: wp.vec3,                     # Camera position (3,)
    clamped_state: wp.array(dtype=wp.vec3), # Clamping state {0,1} from forward pass (N, 3)
    dL_dcolor: wp.array(dtype=wp.vec3),   # Grad L w.r.t. *final* gaussian color (N, 3)

    # --- Outputs (Accumulate) ---
    dL_dmeans_global: wp.array(dtype=wp.vec3), # Accumulate mean grads here (N, 3)
    dL_dshs_global: wp.array(dtype=wp.vec3)   # Accumulate SH grads here (N * 16, 3)
):
    idx = wp.tid()
    if idx >= num_points or radii[idx] <= 0: # Skip if not rendered
        return

    mean = means[idx]
    base_sh_idx = idx * 16

    # --- Recompute view direction ---
    dir_orig = mean - campos
    dir_len = wp.length(dir_orig)
    dir = wp.vec3(0.0, 0.0, 0.0)
    is_zero_len = (dir_len < 1e-8)
    if not is_zero_len:
        dir = dir_orig / dir_len

    x = dir[0]; y = dir[1]; z = dir[2]

    # --- Apply clamping mask to input gradient ---
    dL_dRawColor = dL_dcolor[idx] * (wp.vec3(1.0, 1.0, 1.0) - clamped_state[idx])

    # Initialize local gradients for this Gaussian
    dL_dmeans_local = wp.vec3(0.0, 0.0, 0.0)
    dL_dshs_local = wp.zeros(shape=(16,), dtype=wp.vec3)

    # Initialize gradients w.r.t. direction components (dRawColor/ddir)
    dRGBdx = wp.vec3(0.0, 0.0, 0.0)
    dRGBdy = wp.vec3(0.0, 0.0, 0.0)
    dRGBdz = wp.vec3(0.0, 0.0, 0.0)

    # --- Degree 0 ---
    dL_dshs_local[0] = SH_C0 * dL_dRawColor

    # --- Degree 1 ---
    if degree > 0:
        sh1 = shs[base_sh_idx + 1]
        sh2 = shs[base_sh_idx + 2]
        sh3 = shs[base_sh_idx + 3]
        dL_dshs_local[1] = (-SH_C1 * y) * dL_dRawColor
        dL_dshs_local[2] = ( SH_C1 * z) * dL_dRawColor
        dL_dshs_local[3] = (-SH_C1 * x) * dL_dRawColor

        dRGBdx = dRGBdx - SH_C1 * sh3
        dRGBdy = dRGBdy - SH_C1 * sh1
        dRGBdz = dRGBdz + SH_C1 * sh2

    # --- Degree 2 ---
    if degree > 1:
        xx = x*x; yy = y*y; zz = z*z
        xy = x*y; yz = y*z; xz = x*z

        sh4 = shs[base_sh_idx + 4]; sh5 = shs[base_sh_idx + 5]
        sh6 = shs[base_sh_idx + 6]; sh7 = shs[base_sh_idx + 7]
        sh8 = shs[base_sh_idx + 8]

        # Hardcoded C2 values
        C2_0 = 1.0925484305920792
        C2_1 = -1.0925484305920792
        C2_2 = 0.31539156525252005
        C2_3 = -1.0925484305920792
        C2_4 = 0.5462742152960396

        dL_dshs_local[4] = (C2_0 * xy) * dL_dRawColor
        dL_dshs_local[5] = (C2_1 * yz) * dL_dRawColor
        dL_dshs_local[6] = (C2_2 * (2.0 * zz - xx - yy)) * dL_dRawColor
        dL_dshs_local[7] = (C2_3 * xz) * dL_dRawColor
        dL_dshs_local[8] = (C2_4 * (xx - yy)) * dL_dRawColor

        # Accumulate dRawColor/ddir components using hardcoded C2
        dRGBdx = dRGBdx + C2_0 * y * sh4 + C2_2 * (-2.0 * x) * sh6 + \
                           C2_3 * z * sh7 + C2_4 * ( 2.0 * x) * sh8
        dRGBdy = dRGBdy + C2_0 * x * sh4 + C2_1 * z * sh5 + \
                           C2_2 * (-2.0 * y) * sh6 + C2_4 * (-2.0 * y) * sh8
        dRGBdz = dRGBdz + C2_1 * y * sh5 + C2_2 * (4.0 * z) * sh6 + \
                           C2_3 * x * sh7

    # --- Degree 3 ---
    if degree > 2:
        sh9 = shs[base_sh_idx + 9]; sh10 = shs[base_sh_idx + 10]
        sh11 = shs[base_sh_idx + 11]; sh12 = shs[base_sh_idx + 12]
        sh13 = shs[base_sh_idx + 13]; sh14 = shs[base_sh_idx + 14]
        sh15 = shs[base_sh_idx + 15]

        # Hardcoded C3 values
        C3_0 = -0.5900435899266435
        C3_1 = 2.890611442640554
        C3_2 = -0.4570457994644658
        C3_3 = 0.3731763325901154
        C3_4 = -0.4570457994644658
        C3_5 = 1.445305721320277
        C3_6 = -0.5900435899266435

        dL_dshs_local[9] = (C3_0 * y * (3.0 * xx - yy)) * dL_dRawColor
        dL_dshs_local[10] = (C3_1 * xy * z) * dL_dRawColor
        dL_dshs_local[11] = (C3_2 * y * (4.0 * zz - xx - yy)) * dL_dRawColor
        dL_dshs_local[12] = (C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) * dL_dRawColor
        dL_dshs_local[13] = (C3_4 * x * (4.0 * zz - xx - yy)) * dL_dRawColor
        dL_dshs_local[14] = (C3_5 * z * (xx - yy)) * dL_dRawColor
        dL_dshs_local[15] = (C3_6 * x * (xx - 3.0 * yy)) * dL_dRawColor

        # Accumulate dRawColor/ddir components using hardcoded C3
        term9_dx = y * (6.0 * x); term10_dx = y * z; term11_dx = y * (-2.0 * x)
        term12_dx = z * (-6.0 * x); term13_dx = (4.0*zz - xx - yy) + x * (-2.0*x)
        term14_dx = z * (2.0 * x); term15_dx = (xx - 3.0*yy) + x * (2.0*x)
        dRGBdx = dRGBdx + (C3_0 * sh9 * term9_dx + C3_1 * sh10 * term10_dx +
                           C3_2 * sh11 * term11_dx + C3_3 * sh12 * term12_dx +
                           C3_4 * sh13 * term13_dx + C3_5 * sh14 * term14_dx +
                           C3_6 * sh15 * term15_dx)

        term9_dy = (3.0*xx - yy) + y * (-2.0*y); term10_dy = x * z
        term11_dy = (4.0*zz - xx - yy) + y * (-2.0*y); term12_dy = z * (-6.0 * y)
        term13_dy = x * (-2.0 * y); term14_dy = z * (-2.0 * y); term15_dy = x * (-6.0 * y)
        dRGBdy = dRGBdy + (C3_0 * sh9 * term9_dy + C3_1 * sh10 * term10_dy +
                           C3_2 * sh11 * term11_dy + C3_3 * sh12 * term12_dy +
                           C3_4 * sh13 * term13_dy + C3_5 * sh14 * term14_dy +
                           C3_6 * sh15 * term15_dy)

        term10_dz = x * y; term11_dz = y * (8.0 * z)
        term12_dz = (2.0*zz - 3.0*xx - 3.0*yy) + z * (4.0*z)
        term13_dz = x * (8.0 * z); term14_dz = (xx - yy)
        dRGBdz = dRGBdz + (C3_1 * sh10 * term10_dz + C3_2 * sh11 * term11_dz +
                           C3_3 * sh12 * term12_dz + C3_4 * sh13 * term13_dz +
                           C3_5 * sh14 * term14_dz)

    # --- Compute gradient w.r.t. view direction (dL/ddir) ---
    dL_ddir = wp.vec3(wp.dot(dRGBdx, dL_dRawColor),
                      wp.dot(dRGBdy, dL_dRawColor),
                      wp.dot(dRGBdz, dL_dRawColor))

    # --- Propagate gradient from direction to mean position (dL/dmean) ---
    if not is_zero_len:
        dL_dmeans_local = gradient_normalize_wp(dir_orig, dL_ddir)

    # --- Accumulate gradients to global arrays ---
    wp.atomic_add(dL_dmeans_global, idx, dL_dmeans_local)
    for i in range(16):
        if (i == 0) or \
           (i < 4 and degree > 0) or \
           (i < 9 and degree > 1) or \
           (i < 16 and degree > 2):
             wp.atomic_add(dL_dshs_global, base_sh_idx + i, dL_dshs_local[i])


@wp.kernel
def compute_cov2d_backward_kernel(
    # --- Inputs ---
    num_points: int,
    means: wp.array(dtype=wp.vec3),         # (N, 3)
    cov3Ds: wp.array(dtype=VEC6),           # Packed 3D cov (N, 6)
    radii: wp.array(dtype=int),             # Radii computed in forward (N,) - used for skipping
    h_x: float, h_y: float,                 # Focal lengths
    tan_fovx: float, tan_fovy: float,
    view_matrix: wp.mat44,                  # World->View matrix (4, 4)
    dL_dconics: wp.array(dtype=wp.vec3),    # Grad L w.r.t. conic (a, b, c) (N, 3)

    # --- Outputs (Accumulate/Write) ---
    dL_dmeans_global: wp.array(dtype=wp.vec3), # Accumulate mean grads here (N, 3)
    dL_dcov3Ds_global: wp.array(dtype=VEC6)   # Write 3D cov grads here (N, 6)
):
    idx = wp.tid()
    if idx >= num_points or radii[idx] <= 0: # Skip if not rendered
        # Still need to zero out dL_dcov3Ds? Yes, otherwise it holds old values.
        dL_dcov3Ds_global[idx] = VEC6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return

    # --- Recompute intermediates from forward pass ---
    mean = means[idx]
    cov3D_packed = cov3Ds[idx] # VEC6

    # 1. Transform means to view space 't'
    # t = wp.transform_point(view_matrix, mean) # This uses perspective divide, we need affine transform
    t = wp.transform(view_matrix, mean) # Affine transform

    # 2. View space clamping logic (check original CUDA for exact logic)
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    tz = t[2]
    # Need to handle tz <= 0 case (though radii check might cover this)
    if tz <= 0.0:
        dL_dcov3Ds_global[idx] = VEC6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return # Cannot compute gradient if point is behind camera

    inv_tz = 1.0 / tz
    txtz = t[0] * inv_tz
    tytz = t[1] * inv_tz

    x_clamped_flag = (txtz < -limx) or (txtz > limx)
    y_clamped_flag = (tytz < -limy) or (tytz > limy)
    x_grad_mul = 1.0 - float(x_clamped_flag) # 1.0 if not clamped, 0.0 if clamped
    y_grad_mul = 1.0 - float(y_clamped_flag)

    # Use clamped position for Jacobian calculation
    tx = wp.min(limx, wp.max(-limx, txtz)) * tz
    ty = wp.min(limy, wp.max(-limy, tytz)) * tz
    # tz remains t[2]

    inv_tz2 = inv_tz * inv_tz
    inv_tz3 = inv_tz2 * inv_tz

    # 3. Jacobian J (non-zero elements)
    J00 = h_x * inv_tz
    J11 = h_y * inv_tz
    J02 = -h_x * tx * inv_tz2
    J12 = -h_y * ty * inv_tz2

    # 4. View rotation W (upper 3x3 of view_matrix)
    W = wp.mat33(view_matrix[0,0], view_matrix[0,1], view_matrix[0,2],
                 view_matrix[1,0], view_matrix[1,1], view_matrix[1,2],
                 view_matrix[2,0], view_matrix[2,1], view_matrix[2,2])

    # 5. Transformation T = J @ W (Note: Original code used T = J * W, verify convention)
    # Let's follow the PyTorch T = W @ J derivation -> T[r,c] = sum_k W[r,k] * J[k,c]
    # We computed T elements based on this previously, let's reuse
    T00 = W[0,0]*J00; T01 = W[0,1]*J11; T02 = W[0,0]*J02 + W[0,1]*J12
    T10 = W[1,0]*J00; T11 = W[1,1]*J11; T12 = W[1,0]*J02 + W[1,1]*J12
    T20 = W[2,0]*J00; T21 = W[2,1]*J11; T22 = W[2,0]*J02 + W[2,1]*J12
    T_full = wp.mat33(T00, T10, T20, T01, T11, T21, T02, T12, T22) # Column major constructor

    # 6. Vrk (3D covariance matrix from packed)
    c0 = cov3D_packed[0]; c1 = cov3D_packed[1]; c2 = cov3D_packed[2]
    c11 = cov3D_packed[3]; c12 = cov3D_packed[4]; c22 = cov3D_packed[5]
    Vrk = wp.mat33(c0, c1, c2, c1, c11, c12, c2, c12, c22) # Assumes VEC6 stores upper triangle row-wise

    # 7. Recompute 2D Covariance cov2D = T @ Vrk @ T.T (Matching PyTorch version now)
    cov2D_mat = T_full * Vrk * wp.transpose(T_full)

    a_noblr = cov2D_mat[0,0]; b_noblr = cov2D_mat[0,1]; c_noblr = cov2D_mat[1,1]
    a = a_noblr + 0.3; b = b_noblr; c = c_noblr + 0.3 # Add blur

    # 8. Denominator for conic inversion gradient
    denom = a * c - b * b
    dL_da = 0.0; dL_db = 0.0; dL_dc = 0.0

    # --- Calculate Gradients ---
    if denom != 0.0:
        denom2inv = 1.0 / (denom * denom)
        dL_dconic = dL_dconics[idx] # Input grad w.r.t (a, b, c) of INVERSE conic
        # Need grad w.r.t (a, b, c) of FORWARD conic cov2D_mat
        # Let Conic = [[A, B], [B, C]] = inv(Cov2D)
        # Cov2D = [[a, b], [b, c]]
        # A = c/det; B = -b/det; C = a/det
        # dL/da = dL/dA * dA/da + dL/dB * dB/da + dL/dC * dC/da
        # dA/da = -c*c/det^2 * d(det)/da = -A * c/det * d(det)/da -- complex chain rule
        # Instead, let's use the formula from the CUDA source which computed dL/da, dL/db, dL/dc directly
        # assuming dL_dconic is grad w.r.t Conic = [c/det, -b/det, a/det]
        # So dL_dconic input should be grad w.r.t A, B, C
        dL_dA = dL_dconic[0]; dL_dB = dL_dconic[1]; dL_dC = dL_dconic[2]

        # Chain rule back to a,b,c:
        # dL/da = dL/dA * (dA/da) + dL/dB * (dB/da) + dL/dC * (dC/da)
        # dA/da = -c*C/det; dB/da = b*C/det; dC/da = C*C/det + 1/det
        # dL/da = dL_dA*(-c*C/det) + dL_dB*(b*C/det) + dL_dC*(C*C/det + 1/det) -> Seems different from CUDA impl.

        # Let's trust the CUDA formula derivation implies dL_dconics is grad w.r.t forward a,b,c
        # but computed via the inverse conic gradient dL_d(inv_conic) from rasterizer backward pass.
        # Assuming dL_dconics IS dL/da, dL/db, dL/dc:
        dL_da = dL_dconic[0]; dL_db = dL_dconic[1]; dL_dc = dL_dconic[2]
        # If this assumption is wrong, the grad calc below needs adjustment.

    # Gradients w.r.t packed 3D covariance (dL_dcov3Ds)
    # dL/dVrk = T.T @ dL/dCov2D @ T
    # dL/dCov2D = [[dL/da, 0.5*dL/db], [0.5*dL/db, dL/dc]]
    dL_dCov2D = wp.mat33(dL_da, 0.5*dL_db, 0.0, 0.5*dL_db, dL_dc, 0.0, 0.0, 0.0, 0.0)
    dL_dVrk_mat = wp.transpose(T_full) * dL_dCov2D * T_full

    # Extract packed gradients (factor of 2 for off-diagonals due to symmetry)
    dL_dc00 = dL_dVrk_mat[0,0]
    dL_dc11 = dL_dVrk_mat[1,1]
    dL_dc22 = dL_dVrk_mat[2,2]
    dL_dc01 = 2.0 * dL_dVrk_mat[0,1]
    dL_dc02 = 2.0 * dL_dVrk_mat[0,2]
    dL_dc12 = 2.0 * dL_dVrk_mat[1,2]
    dL_dcov3Ds_global[idx] = VEC6(dL_dc00, dL_dc01, dL_dc02, dL_dc11, dL_dc12, dL_dc22)

    # Gradients w.r.t. T matrix elements (dL_dT)
    # dL/dT = dL/dCov2D @ T @ Vrk + dL/dCov2D.T @ T @ Vrk
    # Since dL/dCov2D is symmetric, dL/dT = 2.0 * dL/dCov2D @ T @ Vrk
    dL_dT_mat = 2.0 * dL_dCov2D * T_full * Vrk

    # Gradients w.r.t. Jacobian non-zero elements (dL_dJ) using T = W @ J => dL/dJ = W.T @ dL/dT
    dL_dJ_mat = wp.transpose(W) * dL_dT_mat
    dL_dJ00 = dL_dJ_mat[0,0]; dL_dJ11 = dL_dJ_mat[1,1]
    dL_dJ02 = dL_dJ_mat[2,0]; dL_dJ12 = dL_dJ_mat[2,1] # Check indices J[k,c]

    # Gradients w.r.t. view-space mean t = (tx, ty, tz)
    dL_dtx = -h_x * inv_tz2 * dL_dJ02
    dL_dty = -h_y * inv_tz2 * dL_dJ12
    dL_dtz = (-h_x * inv_tz2 * dL_dJ00) + \
             (-h_y * inv_tz2 * dL_dJ11) + \
             (2.0 * h_x * tx * inv_tz3 * dL_dJ02) + \
             (2.0 * h_y * ty * inv_tz3 * dL_dJ12)

    dL_dt = wp.vec3(dL_dtx, dL_dty, dL_dtz)

    # Apply clamping mask
    dL_dt = wp.vec3(dL_dt[0] * x_grad_mul, dL_dt[1] * y_grad_mul, dL_dt[2])

    # Gradient w.r.t world mean (transform gradient back using wp.transform_vector)
    dL_dmean_from_cov = wp.transform_vector(view_matrix, dL_dt) # Use built-in

    # Accumulate gradient w.r.t means
    wp.atomic_add(dL_dmeans_global, idx, dL_dmean_from_cov)


@wp.kernel
def compute_cov3d_backward_kernel(
    # --- Inputs ---
    num_points: int,
    scales: wp.array(dtype=wp.vec3),    # (N, 3)
    rotations: wp.array(dtype=wp.vec4), # Quaternions (x, y, z, w) (N, 4)
    radii: wp.array(dtype=int),         # Radii computed in forward (N,) - used for skipping
    scale_modifier: float,
    dL_dcov3Ds: wp.array(dtype=VEC6),   # Grad L w.r.t packed 3D cov (N, 6)

    # --- Outputs ---
    dL_dscales_global: wp.array(dtype=wp.vec3), # Write scale grads here (N, 3)
    dL_drots_global: wp.array(dtype=wp.vec4)   # Write rot grads here (N, 4)
):
    idx = wp.tid()
    # Skip if not rendered OR if grad input is zero (e.g., from compute_cov2d_backward)
    if idx >= num_points or radii[idx] <= 0:
        dL_dscales_global[idx] = wp.vec3(0.0, 0.0, 0.0)
        dL_drots_global[idx] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        return

    # --- Recompute intermediates ---
    scale_vec = scales[idx]
    rot_quat = rotations[idx] # (x, y, z, w)

    # 1. Rotation matrix R from normalized quaternion
    # Must normalize here for derivative calculation w.r.t. normalized quat
    rot_quat_norm = wp.normalize(rot_quat)
    R = quat_to_mat33_wp(rot_quat_norm) # Use the helper

    # 2. Scaling matrix S
    s_vec = scale_modifier * scale_vec
    S = wp.mat33(s_vec[0], 0.0, 0.0, 0.0, s_vec[1], 0.0, 0.0, 0.0, s_vec[2])

    # 3. M = R @ S
    M = R * S # Warp matrix multiplication

    # --- Convert dL_dcov3D (packed) to dL_dSigma (symmetric matrix) ---
    dL_dcov3D_packed = dL_dcov3Ds[idx]
    dL_dc0 = dL_dcov3D_packed[0]; dL_dc1 = dL_dcov3D_packed[1]; dL_dc2 = dL_dcov3D_packed[2]
    dL_dc11 = dL_dcov3D_packed[3]; dL_dc12 = dL_dcov3D_packed[4]; dL_dc22 = dL_dcov3D_packed[5]
    dL_dSigma = wp.mat33(dL_dc0, 0.5 * dL_dc1, 0.5 * dL_dc2,
                         0.5 * dL_dc1, dL_dc11, 0.5 * dL_dc12,
                         0.5 * dL_dc2, 0.5 * dL_dc12, dL_dc22)

    # --- Calculate Gradients ---
    # 1. Gradient w.r.t. M: dL/dM = 2 * dL/dSigma @ M.T (using formula from C++)
    dL_dM = 2.0 * dL_dSigma * wp.transpose(M)

    # 2. Gradient w.r.t. S: dL/dS = R.T @ dL/dM
    dL_dS = wp.transpose(R) * dL_dM

    # 3. Gradient w.r.t. scales: Extract diagonal of dL/dS and apply chain rule
    dL_dscale_vec = wp.vec3(dL_dS[0,0], dL_dS[1,1], dL_dS[2,2]) * scale_modifier
    dL_dscales_global[idx] = dL_dscale_vec

    # 4. Gradient w.r.t. R: dL/dR = dL/dM @ S
    dL_dR = dL_dM * S

    # 5. Gradient w.r.t. quaternion q (x,y,z,w) from dL/dR
    # Use formula for dL/dq (gradient w.r.t. *normalized* quat) based on dL_dM.T
    dL_dMt = wp.transpose(dL_dM)

    # Normalized quaternion components (already computed as rot_quat_norm)
    x = rot_quat_norm[0]; y = rot_quat_norm[1]; z = rot_quat_norm[2]; r = rot_quat_norm[3] # w=r

    dL_dq_r = 2.0 * (z * (dL_dMt[0, 1] - dL_dMt[1, 0]) + \
                     y * (dL_dMt[2, 0] - dL_dMt[0, 2]) + \
                     x * (dL_dMt[1, 2] - dL_dMt[2, 1]))

    dL_dq_x = 2.0 * (y * (dL_dMt[1, 0] + dL_dMt[0, 1]) + \
                     z * (dL_dMt[2, 0] + dL_dMt[0, 2]) + \
                     r * (dL_dMt[1, 2] - dL_dMt[2, 1])) - \
                     4.0 * x * (dL_dMt[1, 1] + dL_dMt[2, 2]) # Corrected index

    dL_dq_y = 2.0 * (x * (dL_dMt[1, 0] + dL_dMt[0, 1]) + \
                     r * (dL_dMt[2, 0] - dL_dMt[0, 2]) + \
                     z * (dL_dMt[1, 2] + dL_dMt[2, 1])) - \
                     4.0 * y * (dL_dMt[0, 0] + dL_dMt[2, 2]) # Corrected index

    dL_dq_z = 2.0 * (r * (dL_dMt[0, 1] - dL_dMt[1, 0]) + \
                     x * (dL_dMt[2, 0] + dL_dMt[0, 2]) + \
                     y * (dL_dMt[1, 2] + dL_dMt[2, 1])) - \
                     4.0 * z * (dL_dMt[0, 0] + dL_dMt[1, 1]) # Corrected index

    # Gradient w.r.t normalized quaternion (order x, y, z, w)
    dL_dquat_normalized = wp.vec4(dL_dq_x, dL_dq_y, dL_dq_z, dL_dq_r)

    # 6. Apply gradient of normalization (chain rule) back to original unnormalized quat
    # dL/drot = dL/dquat_normalized * dquat_normalized/drot
    # Use the helper function gradient_normalize_quat_wp
    dL_drot_unnormalized = gradient_normalize_quat_wp(rot_quat, dL_dquat_normalized)
    dL_drots_global[idx] = dL_drot_unnormalized


@wp.kernel
def compute_projection_backward_kernel(
    # --- Inputs ---
    num_points: int,
    means: wp.array(dtype=wp.vec3),        # (N, 3)
    radii: wp.array(dtype=int),             # Radii computed in forward (N,) - used for skipping
    view_matrix: wp.mat44,                  # Need view matrix for p_view calc
    proj_matrix: wp.mat44,                  # Projection matrix (4, 4)
    dL_dmean2D: wp.array(dtype=wp.vec2),    # Grad L w.r.t. NDC coords (N, 2)
    dL_dDepth: wp.array(dtype=float),       # Grad L w.r.t. View Space Depth (N,)

    # --- Outputs ---
    dL_dmeans_global: wp.array(dtype=wp.vec3) # Accumulate mean grads here (N, 3)
):
    idx = wp.tid()
    if idx >= num_points or radii[idx] <= 0: # Skip if not rendered
        return

    mean = means[idx]

    # --- Recompute forward projection ---
    # Need view space point p_view and clip space point clip_coords
    p_view = wp.transform(view_matrix, mean) # affine transform mean -> p_view
    p_view_h = wp.vec4(p_view[0], p_view[1], p_view[2], 1.0)
    clip_coords = proj_matrix * p_view_h # projection p_view_h -> clip_coords

    w = clip_coords[3]
    # Handle case where w is zero or very small (point at infinity/camera center)
    if wp.abs(w) < 1e-8:
        return # Cannot compute perspective divide gradient

    inv_w = 1.0 / w
    inv_w2 = inv_w * inv_w

    # NDC coordinates (needed for gradient calculation)
    ndc_x = clip_coords[0] * inv_w
    ndc_y = clip_coords[1] * inv_w
    # ndc_z = clip_coords[2] * inv_w # Clip space Z / w

    # --- Calculate Gradients ---
    # Input gradients
    dL_dndc_xy = dL_dmean2D[idx]
    dL_dview_z = dL_dDepth[idx] # Gradient w.r.t p_view.z

    # 1. Gradient w.r.t. Clip Coordinates (dL_dClip)
    # Backpropagate dL/dNdc_xy to dL/dClip (x, y, w components)
    dL_dclip_x = dL_dndc_xy[0] * inv_w
    dL_dclip_y = dL_dndc_xy[1] * inv_w
    dL_dclip_z = 0.0 # Assuming dL/dNdc_z = 0 for now
    dL_dclip_w = (dL_dndc_xy[0] * (-ndc_x * inv_w)) + \
                 (dL_dndc_xy[1] * (-ndc_y * inv_w)) # + dL/dNdc_z * (-ndc_z * inv_w) if needed

    dL_dclip = wp.vec4(dL_dclip_x, dL_dclip_y, dL_dclip_z, dL_dclip_w)

    # 2. Gradient w.r.t. View Space Coordinates (dL_dp_view_h)
    # dL/dp_view_h = dL/dClip * dClip/dp_view_h = dL/dClip * Proj.T
    # Calculate dL_dClip (1x4) @ Proj (4x4) using helper
    # dL_dp_view_h_j = sum_i dL_dclip_i * Proj[i, j]
    dL_dp_view_h = wp.vec4(
        wp.dot(dL_dclip, proj_matrix[0]), # dL/dp_view.x
        wp.dot(dL_dclip, proj_matrix[1]), # dL/dp_view.y
        wp.dot(dL_dclip, proj_matrix[2]), # dL/dp_view.z (from projection)
        wp.dot(dL_dclip, proj_matrix[3])  # dL/dp_view.w (ignore for now)
    )

    # Add gradient contribution from depth (dL/dview_z) directly to z component
    dL_dp_view = wp.vec3(dL_dp_view_h[0], dL_dp_view_h[1], dL_dp_view_h[2] + dL_dview_z)

    # 3. Gradient w.r.t World Space Mean (dL_dmean)
    # dL/dmean = dL/dp_view * dp_view/dmean = dL/dp_view * View[:3,:3].T
    dL_dmean_from_proj = wp.transform_vector(view_matrix, dL_dp_view) # Use built-in

    # Accumulate gradient w.r.t means
    wp.atomic_add(dL_dmeans_global, idx, dL_dmean_from_proj)