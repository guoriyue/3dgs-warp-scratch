import warp as wp
import math
from utils.wp_utils import to_warp_array, wp_vec3_mul_element, wp_vec3_add_element, wp_vec3_sqrt, wp_vec3_div_element
from config import * # Assuming TILE_M, TILE_N, VEC6, DEVICE are defined here

# Initialize Warp if not already done elsewhere
# wp.init()

# --- Spherical Harmonics Constants ---
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199

@wp.func
def dnormvdv(v: wp.vec3, dv: wp.vec3) -> wp.vec3:
    """
    Computes the gradient of normalize(v) with respect to v, scaled by dv.
    This is a direct port of the CUDA implementation.
    
    Args:
        v: The input vector to be normalized
        dv: The gradient vector to scale the result by
        
    Returns:
        The gradient vector
    """
    sum2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    
    # Avoid division by zero
    if sum2 < 1e-10:
        return wp.vec3(0.0, 0.0, 0.0)
        
    invsum32 = 1.0 / wp.sqrt(sum2 * sum2 * sum2)
    
    result = wp.vec3(
        ((sum2 - v[0] * v[0]) * dv[0] - v[1] * v[0] * dv[1] - v[2] * v[0] * dv[2]) * invsum32,
        (-v[0] * v[1] * dv[0] + (sum2 - v[1] * v[1]) * dv[1] - v[2] * v[1] * dv[2]) * invsum32,
        (-v[0] * v[2] * dv[0] - v[1] * v[2] * dv[1] + (sum2 - v[2] * v[2]) * dv[2]) * invsum32
    )
    
    return result

# --- Backward Kernels ---
@wp.kernel
def sh_backward_kernel(
    # --- Inputs ---
    num_points: int,
    degree: int, # SH degree used in forward
    means: wp.array(dtype=wp.vec3),      # (N, 3)
    shs: wp.array(dtype=wp.vec3),        # Flattened SH coeffs (N * 16, 3)
    radii: wp.array(dtype=int),          # Radii computed in forward (N,) - used for skipping
    campos: wp.vec3,                     # Camera position (3,)
    clamped_state: wp.array(dtype=wp.vec3), # Clamping state {0,1} from forward pass (N, 3)
    dL_dcolor: wp.array(dtype=wp.vec3),   # Grad L w.r.t. *final* gaussian color (N, 3)

    # --- Outputs (Accumulate) ---
    dL_dmeans: wp.array(dtype=wp.vec3), # Accumulate mean grads here (N, 3)
    dL_dshs: wp.array(dtype=wp.vec3)   # Accumulate SH grads here (N * 16, 3)
):
    idx = wp.tid()
    
    if idx >= num_points or radii[idx] <= 0: # Skip if not rendered
        return

    mean = means[idx]
    base_sh_idx = idx * 16

    # --- Recompute view direction ---
    dir_orig = mean - campos
    dir_len = wp.length(dir_orig)
    # Skip if direction length is too small (matches CUDA implementation)
    if dir_len < 1e-8:
        return
        
    # Normalize direction
    dir = dir_orig / dir_len
    x = dir[0]; y = dir[1]; z = dir[2]

    # --- Apply clamping mask to input gradient ---
    # In CUDA: dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
    # Here we use: 1.0 - clamped_state which gives 0 for clamped, 1 for not clamped
    dL_dRGB = dL_dcolor[idx]
    dL_dRGB = wp_vec3_mul_element(dL_dRGB, wp_vec3_add_element(wp.vec3(1.0, 1.0, 1.0), -1.0 * clamped_state[idx]))

    # Initialize gradients w.r.t. direction components (dRawColor/ddir)
    dRGBdx = wp.vec3(0.0, 0.0, 0.0)
    dRGBdy = wp.vec3(0.0, 0.0, 0.0)
    dRGBdz = wp.vec3(0.0, 0.0, 0.0)

    # --- Degree 0 ---
    # Direct assignment for clarity (matching CUDA style)
    dRGBdsh0 = SH_C0
    dL_dshs[base_sh_idx] = dRGBdsh0 * dL_dRGB

    # --- Degree 1 ---
    if degree > 0:
        sh1 = shs[base_sh_idx + 1]
        sh2 = shs[base_sh_idx + 2]
        sh3 = shs[base_sh_idx + 3]
        
        # Exactly match CUDA computation order
        dRGBdsh1 = -SH_C1 * y
        dRGBdsh2 = SH_C1 * z
        dRGBdsh3 = -SH_C1 * x
        
        dL_dshs[base_sh_idx + 1] = dRGBdsh1 * dL_dRGB
        dL_dshs[base_sh_idx + 2] = dRGBdsh2 * dL_dRGB
        dL_dshs[base_sh_idx + 3] = dRGBdsh3 * dL_dRGB

        # Gradient components w.r.t. direction
        dRGBdx = -SH_C1 * sh3
        dRGBdy = -SH_C1 * sh1
        dRGBdz = SH_C1 * sh2
        # --- Degree 2 ---
        if degree > 1:
            xx = x*x; yy = y*y; zz = z*z
            xy = x*y; yz = y*z; xz = x*z

            sh4 = shs[base_sh_idx + 4]; sh5 = shs[base_sh_idx + 5]
            sh6 = shs[base_sh_idx + 6]; sh7 = shs[base_sh_idx + 7]
            sh8 = shs[base_sh_idx + 8]

            # Hardcoded C2 values (same as CUDA SH_C2)
            C2_0 = 1.0925484305920792
            C2_1 = -1.0925484305920792
            C2_2 = 0.31539156525252005
            C2_3 = -1.0925484305920792
            C2_4 = 0.5462742152960396

            # Compute gradients for degree 2 (matching CUDA)
            dRGBdsh4 = C2_0 * xy
            dRGBdsh5 = C2_1 * yz
            dRGBdsh6 = C2_2 * (2.0 * zz - xx - yy)
            dRGBdsh7 = C2_3 * xz
            dRGBdsh8 = C2_4 * (xx - yy)
            
            dL_dshs[base_sh_idx + 4] = dRGBdsh4 * dL_dRGB
            dL_dshs[base_sh_idx + 5] = dRGBdsh5 * dL_dRGB
            dL_dshs[base_sh_idx + 6] = dRGBdsh6 * dL_dRGB
            dL_dshs[base_sh_idx + 7] = dRGBdsh7 * dL_dRGB
            dL_dshs[base_sh_idx + 8] = dRGBdsh8 * dL_dRGB

            # Accumulate gradients w.r.t. direction (exactly matching CUDA)
            dRGBdx += C2_0 * y * sh4 + C2_2 * 2.0 * -x * sh6 + C2_3 * z * sh7 + C2_4 * 2.0 * x * sh8
            dRGBdy += C2_0 * x * sh4 + C2_1 * z * sh5 + C2_2 * 2.0 * -y * sh6 + C2_4 * 2.0 * -y * sh8
            dRGBdz += C2_1 * y * sh5 + C2_2 * 2.0 * 2.0 * z * sh6 + C2_3 * x * sh7

            # --- Degree 3 ---
            if degree > 2:
                sh9 = shs[base_sh_idx + 9]; sh10 = shs[base_sh_idx + 10]
                sh11 = shs[base_sh_idx + 11]; sh12 = shs[base_sh_idx + 12]
                sh13 = shs[base_sh_idx + 13]; sh14 = shs[base_sh_idx + 14]
                sh15 = shs[base_sh_idx + 15]

                # Hardcoded C3 values (same as CUDA SH_C3)
                C3_0 = -0.5900435899266435
                C3_1 = 2.890611442640554
                C3_2 = -0.4570457994644658
                C3_3 = 0.3731763325901154
                C3_4 = -0.4570457994644658
                C3_5 = 1.445305721320277
                C3_6 = -0.5900435899266435

                # Direct computation of degree 3 gradients (matching CUDA)
                dRGBdsh9 = C3_0 * y * (3.0 * xx - yy)
                dRGBdsh10 = C3_1 * xy * z
                dRGBdsh11 = C3_2 * y * (4.0 * zz - xx - yy)
                dRGBdsh12 = C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                dRGBdsh13 = C3_4 * x * (4.0 * zz - xx - yy)
                dRGBdsh14 = C3_5 * z * (xx - yy)
                dRGBdsh15 = C3_6 * x * (xx - 3.0 * yy)
                
                dL_dshs[base_sh_idx + 9] = dRGBdsh9 * dL_dRGB
                dL_dshs[base_sh_idx + 10] = dRGBdsh10 * dL_dRGB
                dL_dshs[base_sh_idx + 11] = dRGBdsh11 * dL_dRGB
                dL_dshs[base_sh_idx + 12] = dRGBdsh12 * dL_dRGB
                dL_dshs[base_sh_idx + 13] = dRGBdsh13 * dL_dRGB
                dL_dshs[base_sh_idx + 14] = dRGBdsh14 * dL_dRGB
                dL_dshs[base_sh_idx + 15] = dRGBdsh15 * dL_dRGB

                # Accumulate dRGBdx (matching CUDA's expression structure)
                dRGBdx += (
                    C3_0 * sh9 * 3.0 * 2.0 * xy +
                    C3_1 * sh10 * yz +
                    C3_2 * sh11 * -2.0 * xy +
                    C3_3 * sh12 * -3.0 * 2.0 * xz +
                    C3_4 * sh13 * (-3.0 * xx + 4.0 * zz - yy) +
                    C3_5 * sh14 * 2.0 * xz +
                    C3_6 * sh15 * 3.0 * (xx - yy)
                )

                # Accumulate dRGBdy (matching CUDA's expression structure)
                dRGBdy += (
                    C3_0 * sh9 * 3.0 * (xx - yy) +
                    C3_1 * sh10 * xz +
                    C3_2 * sh11 * (-3.0 * yy + 4.0 * zz - xx) +
                    C3_3 * sh12 * -3.0 * 2.0 * yz +
                    C3_4 * sh13 * -2.0 * xy +
                    C3_5 * sh14 * -2.0 * yz +
                    C3_6 * sh15 * -3.0 * 2.0 * xy
                )

                # Accumulate dRGBdz (matching CUDA's expression structure)
                dRGBdz += (
                    C3_1 * sh10 * xy +
                    C3_2 * sh11 * 4.0 * 2.0 * yz +
                    C3_3 * sh12 * 3.0 * (2.0 * zz - xx - yy) +
                    C3_4 * sh13 * 4.0 * 2.0 * xz +
                    C3_5 * sh14 * (xx - yy)
                )

    # --- Compute gradient w.r.t. view direction (dL/ddir) ---
    # In CUDA: glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));
    dL_ddir = wp.vec3(wp.dot(dRGBdx, dL_dRGB),
                      wp.dot(dRGBdy, dL_dRGB),
                      wp.dot(dRGBdz, dL_dRGB))

    # --- Propagate gradient from direction to mean position (dL/dmean) ---
    # In CUDA: float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });
    dL_dmeans_local = dnormvdv(dir_orig, dL_ddir)

    # --- Accumulate gradients to global arrays ---
    # In CUDA: dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
    dL_dmeans[idx] += dL_dmeans_local


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
    dL_dconics: wp.array(dtype=wp.vec4),    # Grad L w.r.t. conic (a, b, c) (N, 3)

    # --- Outputs (Accumulate/Write) ---
    dL_dmeans: wp.array(dtype=wp.vec3), # Accumulate mean grads here (N, 3)
    dL_dcov3Ds: wp.array(dtype=VEC6)   # Write 3D cov grads here (N, 6)
):
    idx = wp.tid()
    if idx >= num_points or radii[idx] <= 0: # Skip if not rendered
        # Zero out dL_dcov3Ds to ensure we don't keep old values
        dL_dcov3Ds[idx] = VEC6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return
    
    mean = means[idx]
    cov3D_packed = cov3Ds[idx] # VEC6
    
    dL_dconic = wp.vec3(dL_dconics[idx][0], dL_dconics[idx][1], dL_dconics[idx][3])
    
    if idx == 0:
        # dL_dconic = wp.vec3(0.0, 0.0, 0.0, 0.0)
        dL_dconic = wp.vec3(-0.317618, -0.0099874455, -0.2904277742)
        print(dL_dconic)
        print("XXXX")
        
    t = wp.vec4(mean[0], mean[1], mean[2], 1.0) * view_matrix

    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    tz = t[2]
    inv_tz = 1.0 / tz
    txtz = t[0] * inv_tz
    tytz = t[1] * inv_tz

    x_clamped_flag = (txtz < -limx) or (txtz > limx)
    y_clamped_flag = (tytz < -limy) or (tytz > limy)
    x_grad_mul = 1.0 - float(x_clamped_flag) # 1.0 if not clamped, 0.0 if clamped
    y_grad_mul = 1.0 - float(y_clamped_flag)

    tx = wp.min(limx, wp.max(-limx, txtz)) * tz
    ty = wp.min(limy, wp.max(-limy, tytz)) * tz
    inv_tz2 = inv_tz * inv_tz
    inv_tz3 = inv_tz2 * inv_tz

    J00 = h_x * inv_tz
    J11 = h_y * inv_tz
    J02 = -h_x * tx * inv_tz2
    J12 = -h_y * ty * inv_tz2

    J = wp.transpose(wp.mat33(
        J00, 0.0, J02,
        0.0, J11, J12,
        0.0, 0.0, 0.0
    ))
    
    

    W = wp.mat33(
        view_matrix[0,0], view_matrix[0,1], view_matrix[0,2],
        view_matrix[1,0], view_matrix[1,1], view_matrix[1,2],
        view_matrix[2,0], view_matrix[2,1], view_matrix[2,2]
    )

    T = W * J
    c0 = cov3D_packed[0]; c1 = cov3D_packed[1]; c2 = cov3D_packed[2]
    c11 = cov3D_packed[3]; c12 = cov3D_packed[4]; c22 = cov3D_packed[5]
    Vrk = wp.mat33(c0, c1, c2, c1, c11, c12, c2, c12, c22) # Assumes VEC6 stores upper triangle row-wise
    
    cov2D_mat = wp.transpose(T) * wp.transpose(Vrk) * T

    a_noblr = cov2D_mat[0,0]
    b_noblr = cov2D_mat[0,1] 
    c_noblr = cov2D_mat[1,1]
    a = a_noblr + 0.3
    b = b_noblr
    c = c_noblr + 0.3

    denom = a * c - b * b
    dL_da = 0.0; dL_db = 0.0; dL_dc = 0.0
        
    # --- Calculate Gradients ---
    if denom != 0.0:
        # Use a small epsilon to prevent division by zero
        denom2inv = 1.0 / (denom * denom + 1e-7)
        dL_da = denom2inv * (-c * c * dL_dconic[0] + 2.0 * b * c * dL_dconic[1] + (denom - a * c) * dL_dconic[2])
        dL_dc = denom2inv * (-a * a * dL_dconic[2] + 2.0 * a * b * dL_dconic[1] + (denom - a * c) * dL_dconic[0])
        dL_db = denom2inv * 2.0 * (b * c * dL_dconic[0] - (denom + 2.0 * b * b) * dL_dconic[1] + a * b * dL_dconic[2])

    dL_dcov3Ds[idx] = VEC6(
        # Diagonal elements
        T[0][0] * T[0][0] * dL_da + T[0][0] * T[0][1] * dL_db + T[0][1] * T[0][1] * dL_dc,  # c00
        2.0 * T[0][0] * T[1][0] * dL_da + (T[0][0] * T[1][1] + T[1][0] * T[0][1]) * dL_db + 2.0 * T[0][1] * T[1][1] * dL_dc,  # c01
        2.0 * T[0][0] * T[2][0] * dL_da + (T[0][0] * T[2][1] + T[2][0] * T[0][1]) * dL_db + 2.0 * T[0][1] * T[2][1] * dL_dc,  # c02
        T[1][0] * T[1][0] * dL_da + T[1][0] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc,  # c11
        2.0 * T[2][0] * T[1][0] * dL_da + (T[1][0] * T[2][1] + T[2][0] * T[1][1]) * dL_db + 2.0 * T[1][1] * T[2][1] * dL_dc,  # c12
        T[2][0] * T[2][0] * dL_da + T[2][0] * T[2][1] * dL_db + T[2][1] * T[2][1] * dL_dc   # c22
    )

    dL_dT00 = 2.0 * (T[0][0] * Vrk[0][0] + T[1][0] * Vrk[1][0] + T[2][0] * Vrk[2][0]) * dL_da + \
              (T[0][1] * Vrk[0][0] + T[1][1] * Vrk[1][0] + T[2][1] * Vrk[2][0]) * dL_db
    dL_dT01 = 2.0 * (T[0][0] * Vrk[0][1] + T[1][0] * Vrk[1][1] + T[2][0] * Vrk[2][1]) * dL_da + \
              (T[0][1] * Vrk[0][1] + T[1][1] * Vrk[1][1] + T[2][1] * Vrk[2][1]) * dL_db
    dL_dT02 = 2.0 * (T[0][0] * Vrk[0][2] + T[1][0] * Vrk[1][2] + T[2][0] * Vrk[2][2]) * dL_da + \
              (T[0][1] * Vrk[0][2] + T[1][1] * Vrk[1][2] + T[2][1] * Vrk[2][2]) * dL_db
    dL_dT10 = 2.0 * (T[0][1] * Vrk[0][0] + T[1][1] * Vrk[1][0] + T[2][1] * Vrk[2][0]) * dL_dc + \
              (T[0][0] * Vrk[0][0] + T[1][0] * Vrk[1][0] + T[2][0] * Vrk[2][0]) * dL_db
    dL_dT11 = 2.0 * (T[0][1] * Vrk[0][1] + T[1][1] * Vrk[1][1] + T[2][1] * Vrk[2][1]) * dL_dc + \
              (T[0][0] * Vrk[0][1] + T[1][0] * Vrk[1][1] + T[2][0] * Vrk[2][1]) * dL_db
    dL_dT12 = 2.0 * (T[0][1] * Vrk[0][2] + T[1][1] * Vrk[1][2] + T[2][1] * Vrk[2][2]) * dL_dc + \
              (T[0][0] * Vrk[0][2] + T[1][0] * Vrk[1][2] + T[2][0] * Vrk[2][2]) * dL_db

    dL_dJ00 = W[0,0] * dL_dT00 + W[1,0] * dL_dT01 + W[2,0] * dL_dT02
    dL_dJ02 = W[0,2] * dL_dT00 + W[1,2] * dL_dT01 + W[2,2] * dL_dT02
    dL_dJ11 = W[0,1] * dL_dT10 + W[1,1] * dL_dT11 + W[2,1] * dL_dT12
    dL_dJ12 = W[0,2] * dL_dT10 + W[1,2] * dL_dT11 + W[2,2] * dL_dT12

    dL_dtx = -h_x * inv_tz2 * dL_dJ02
    dL_dty = -h_y * inv_tz2 * dL_dJ12
    dL_dtz = -h_x * inv_tz2 * dL_dJ00 - h_y * inv_tz2 * dL_dJ11 + \
             2.0 * h_x * tx * inv_tz3 * dL_dJ02 + 2.0 * h_y * ty * inv_tz3 * dL_dJ12

    dL_dt = wp.vec3(dL_dtx * x_grad_mul, dL_dty * y_grad_mul, dL_dtz)

    dL_dmean_from_cov = wp.vec4(dL_dt[0], dL_dt[1], dL_dt[2], 1.0) * wp.transpose(view_matrix)
    dL_dmeans[idx] += wp.vec3(dL_dmean_from_cov[0], dL_dmean_from_cov[1], dL_dmean_from_cov[2])


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
    dL_dscales: wp.array(dtype=wp.vec3), # Write scale grads here (N, 3)
    dL_drots: wp.array(dtype=wp.vec4)   # Write rot grads here (N, 4)
):
    idx = wp.tid()
    # Skip if not rendered OR if grad input is zero (e.g., from compute_cov2d_backward)
    if idx >= num_points or radii[idx] <= 0:
        dL_dscales[idx] = wp.vec3(0.0, 0.0, 0.0)
        dL_drots[idx] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        return

    # --- Recompute intermediates ---
    scale_vec = scales[idx]
    rot_quat = rotations[idx] # (x, y, z, w) in Warp

    # Extract quaternion components to match CUDA convention (r, x, y, z)
    # In CUDA: r = q.x, x = q.y, y = q.z, z = q.w
    # In Warp: x = q[0], y = q[1], z = q[2], r = q[3] (w)
    r = rot_quat[3]  # Real part is w in Warp
    x = rot_quat[0]
    y = rot_quat[1]
    z = rot_quat[2]

    # 1. Construct rotation matrix R manually as in CUDA
    # This matches the CUDA implementation exactly
    R = wp.mat33(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
        2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
        2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)
    )

    # 2. Create scaling matrix S
    s_vec = scale_modifier * scale_vec
    S = wp.mat33(
        s_vec[0], 0.0, 0.0,
        0.0, s_vec[1], 0.0,
        0.0, 0.0, s_vec[2]
    )

    # 3. M = S * R (match CUDA multiplication order)
    M = S * R

    # --- Extract gradient w.r.t. 3D covariance ---
    dL_dcov3D_packed = dL_dcov3Ds[idx]
    
    
    # Convert per-element covariance loss gradients to matrix form
    dL_dSigma = wp.mat33(
        dL_dcov3D_packed[0], 0.5 * dL_dcov3D_packed[1], 0.5 * dL_dcov3D_packed[2],
        0.5 * dL_dcov3D_packed[1], dL_dcov3D_packed[3], 0.5 * dL_dcov3D_packed[4],
        0.5 * dL_dcov3D_packed[2], 0.5 * dL_dcov3D_packed[4], dL_dcov3D_packed[5]
    )

    # --- Calculate Gradients ---
    # 1. Gradient w.r.t. M: dL/dM = 2 * M * dL/dSigma
    dL_dM = 2.0 * M * dL_dSigma

    # 2. Transpose of matrices for gradient calculations
    Rt = wp.transpose(R)
    dL_dMt = wp.transpose(dL_dM)

    # 3. Gradient w.r.t. scales - matching CUDA directly
    dL_dscale = wp.vec3(
        wp.dot(Rt[0], dL_dMt[0]),
        wp.dot(Rt[1], dL_dMt[1]), 
        wp.dot(Rt[2], dL_dMt[2])
    )
    dL_dscales[idx] = dL_dscale * scale_modifier

    # 4. Scale dL_dMt by scale factors for quaternion gradient calculation
    dL_dMt_scaled = wp.mat33(
        dL_dMt[0, 0] * s_vec[0], dL_dMt[0, 1] * s_vec[0], dL_dMt[0, 2] * s_vec[0],
        dL_dMt[1, 0] * s_vec[1], dL_dMt[1, 1] * s_vec[1], dL_dMt[1, 2] * s_vec[1],
        dL_dMt[2, 0] * s_vec[2], dL_dMt[2, 1] * s_vec[2], dL_dMt[2, 2] * s_vec[2]
    )

    # 5. Gradients of loss w.r.t. quaternion components
    # Following CUDA implementation exactly with the same indices
    dL_dr = 2.0 * (z * (dL_dMt_scaled[0, 1] - dL_dMt_scaled[1, 0]) + 
                   y * (dL_dMt_scaled[2, 0] - dL_dMt_scaled[0, 2]) + 
                   x * (dL_dMt_scaled[1, 2] - dL_dMt_scaled[2, 1]))
    
    dL_dx = 2.0 * (y * (dL_dMt_scaled[1, 0] + dL_dMt_scaled[0, 1]) + 
                   z * (dL_dMt_scaled[2, 0] + dL_dMt_scaled[0, 2]) + 
                   r * (dL_dMt_scaled[1, 2] - dL_dMt_scaled[2, 1])) - \
            4.0 * x * (dL_dMt_scaled[2, 2] + dL_dMt_scaled[1, 1])
    
    dL_dy = 2.0 * (x * (dL_dMt_scaled[1, 0] + dL_dMt_scaled[0, 1]) + 
                   r * (dL_dMt_scaled[2, 0] - dL_dMt_scaled[0, 2]) + 
                   z * (dL_dMt_scaled[1, 2] + dL_dMt_scaled[2, 1])) - \
            4.0 * y * (dL_dMt_scaled[2, 2] + dL_dMt_scaled[0, 0])
    
    dL_dz = 2.0 * (r * (dL_dMt_scaled[0, 1] - dL_dMt_scaled[1, 0]) + 
                   x * (dL_dMt_scaled[2, 0] + dL_dMt_scaled[0, 2]) + 
                   y * (dL_dMt_scaled[1, 2] + dL_dMt_scaled[2, 1])) - \
            4.0 * z * (dL_dMt_scaled[1, 1] + dL_dMt_scaled[0, 0])

    # 6. Convert back to Warp's quaternion ordering (x, y, z, r/w)
    dL_drots[idx] = wp.vec4(dL_dx, dL_dy, dL_dz, dL_dr)

@wp.kernel
def wp_render_backward_kernel(
    # --- Inputs ---
    # Tile/Range data
    ranges: wp.array(dtype=wp.vec2i),       # Range of point indices for each tile (start, end)
    point_list: wp.array(dtype=int),        # Sorted point indices
    
    # Image parameters
    W: int,                                 # Image width
    H: int,                                 # Image height
    bg_color: wp.vec3,                      # Background color
    tile_grid: wp.vec3,
    
    # Gaussian parameters
    points_xy_image: wp.array(dtype=wp.vec2), # 2D projected positions
    conic_opacity: wp.array(dtype=wp.vec4),   # Conic matrices and opacities (a, b, c, opacity)
    colors: wp.array(dtype=wp.vec3),          # RGB colors
    depths: wp.array(dtype=float),            # Depth values
    
    # Forward pass results
    final_Ts: wp.array2d(dtype=float),      # Final transparency values
    n_contrib: wp.array2d(dtype=int),       # Number of Gaussians contributing to each pixel
    dL_dpixels: wp.array2d(dtype=wp.vec3),  # Gradient of loss w.r.t. output pixels
    dL_invdepths: wp.array2d(dtype=float),  # Gradient of loss w.r.t. inverse depths
    use_invdepth: bool,
    
    # --- Outputs ---
    dL_dmean2D: wp.array(dtype=wp.vec3),    # Gradient w.r.t. 2D mean positions
    dL_dconic2D: wp.array(dtype=wp.vec4),   # Gradient w.r.t. conic matrices
    dL_dopacity: wp.array(dtype=float),     # Gradient w.r.t. opacity
    dL_dcolors: wp.array(dtype=wp.vec3),    # Gradient w.r.t. colors
    dL_dinvdepths: wp.array(dtype=float)    # Gradient w.r.t. inverse depths
):
    """
    Backward version of the rendering procedure, computing gradients of the loss with respect
    to Gaussian parameters based on gradients of the loss with respect to output pixels.
    
    This kernel is launched per pixel and processes Gaussians in back-to-front order,
    similar to the forward rendering pass but accumulating gradients.
    """
    # Get pixel coordinates
    tile_x, tile_y, tid_x, tid_y = wp.tid()
    
    # Calculate pixel position
    pix_x = tile_x * TILE_M + tid_x
    pix_y = tile_y * TILE_N + tid_y
    
    # Skip if pixel is outside image bounds
    inside = (pix_x < W) and (pix_y < H)
    if not inside:
        return
    
    # Convert to float coordinates for calculations
    pixf_x = float(pix_x)
    pixf_y = float(pix_y)
    
    # Get tile range (start/end indices in point_list)
    tile_id = tile_y * int(tile_grid[0]) + tile_x

    range_start = ranges[tile_id][0]
    range_end = ranges[tile_id][1]
        
    # Get final transparency value and number of contributors from forward pass
    T_final = final_Ts[pix_y, pix_x]
    last_contributor = n_contrib[pix_y, pix_x]

    first_kept = max(range_start, range_end - last_contributor)   # = range_end-N

    # Initialize working variables
    T = T_final  # Current accumulated transparency
    accum_rec = wp.vec3(0.0, 0.0, 0.0)  # Accumulated color
    accum_invdepth_rec = float(0.0)  # Accumulated inverse depth
    last_alpha = float(0.0)  # Alpha from the last processed Gaussian
    last_color = wp.vec3(0.0, 0.0, 0.0)  # Color from the last processed Gaussian
    last_invdepth = float(0.0)  # Inverse depth from the last processed Gaussian
    
    # Get gradients
    dL_dpixel = dL_dpixels[pix_y, pix_x]
    # if use_invdepth:
    #     dL_invdepth = dL_invdepths[pix_y, pix_x]
    # else:
    #     dL_invdepth = 0.0
    
    # Gradient of pixel coordinate w.r.t. normalized screen-space coordinates
    ddelx_dx = 0.5 * float(W)
    ddely_dy = 0.5 * float(H)
    # Process Gaussians in back-to-front order
    for i in range(range_end - 1, first_kept  - 1, -1):
        gaussian_id = point_list[i]
        xy = points_xy_image[gaussian_id]
        con_o = conic_opacity[gaussian_id]  # (a, b, c, opacity)
        color = colors[gaussian_id]
        depth = depths[gaussian_id]
        
        # Compute distance to pixel center
        d_x = xy[0] - pixf_x
        d_y = xy[1] - pixf_y
        
        # Compute Gaussian power
        power = -0.5 * (con_o[0] * d_x * d_x + con_o[2] * d_y * d_y) - con_o[1] * d_x * d_y
     
        # Skip if power is positive (too far away)
        if power > 0.0:
            continue
            
        # Compute Gaussian value and alpha
        G = wp.exp(power)
        alpha = wp.min(0.99, con_o[3] * G)

        # Skip if alpha is too small
        if alpha < (1.0 / 255.0):
            continue
            
        T = T / (1.0 - alpha)
        
        # Gradient factor for color contribution
        dchannel_dcolor = alpha * T
        
        # Compute gradient w.r.t. alpha
        dL_dalpha = 0.0
        
        # Update color accumulation and compute color gradients
        accum_rec = last_alpha * last_color + (1.0 - last_alpha) * accum_rec
        dL_dchannel = dL_dpixel
        last_color = color
        
        dL_dalpha = wp.dot(color - accum_rec, dL_dpixel)
        wp.atomic_add(dL_dcolors, gaussian_id, dchannel_dcolor * dL_dchannel)

        # # Handle depth gradients if enabled
        # if use_invdepth:
        #     invd = 1.0 / depth
        #     accum_invdepth_rec = last_alpha * last_invdepth + (1.0 - last_alpha) * accum_invdepth_rec
        #     last_invdepth = invd
        #     dL_dalpha += (invd - accum_invdepth_rec) * dL_invdepth
        #     wp.atomic_add(dL_dinvdepths, gaussian_id, dchannel_dcolor * dL_invdepth)
        
        # Scale dL_dalpha by T
        dL_dalpha *= T
        last_alpha = alpha

        # Account for background color contribution
        bg_dot_dpixel = wp.dot(bg_color, dL_dpixel)
        dL_dalpha += (-T_final / (1.0 - alpha)) * bg_dot_dpixel
        
        # Helpful temporary variables
        dL_dG = con_o[3] * dL_dalpha
        gdx = G * d_x
        gdy = G * d_y
        dG_ddelx = -gdx * con_o[0] - gdy * con_o[1]
        dG_ddely = -gdy * con_o[2] - gdx * con_o[1]

        # Update gradients w.r.t. 2D mean position
        wp.atomic_add(dL_dmean2D, gaussian_id, wp.vec3(
            dL_dG * dG_ddelx * ddelx_dx,
            dL_dG * dG_ddely * ddely_dy,
            0.0
        ))
            
        # Update gradients w.r.t. 2D conic matrix
        wp.atomic_add(dL_dconic2D, gaussian_id, wp.vec4(
            -0.5 * gdx * d_x * dL_dG,
            -0.5 * gdx * d_y * dL_dG,
            0.0,
            -0.5 * gdy * d_y * dL_dG
        ))
        
        # Update gradients w.r.t. opacity
        wp.atomic_add(dL_dopacity, gaussian_id, G * dL_dalpha)

@wp.kernel
def compute_projection_backward_kernel(
    # --- Inputs ---
    num_points: int,
    means: wp.array(dtype=wp.vec3),      # (N, 3) 3D positions
    radii: wp.array(dtype=int),          # Radii computed in forward (N,) - used for skipping
    proj_matrix: wp.mat44,               # Projection matrix (4, 4)
    dL_dmean2D: wp.array(dtype=wp.vec3), # Grad of loss w.r.t. 2D projected means (N, 2)
    
    # --- Outputs (Accumulate) ---
    dL_dmeans: wp.array(dtype=wp.vec3) # Accumulate mean grads here (N, 3)
):
    """Compute gradients of 3D means due to projection to 2D.
    
    This kernel handles the gradient propagation from 2D projected positions
    back to 3D positions, based on the projection matrix.
    """
    idx = wp.tid()
    if idx >= num_points or radii[idx] <= 0: # Skip if not rendered
        return
        
    # Get 3D mean and 2D mean gradient
    mean3D = means[idx]
    dL_dmean2D_val = dL_dmean2D[idx]
    
    # Compute homogeneous coordinates
    m_hom = wp.vec4(mean3D[0], mean3D[1], mean3D[2], 1.0)
    m_hom = m_hom * proj_matrix
    
    # Division by w (perspective division)
    m_w = 1.0 / (m_hom[3] + 0.0000001)
    
    # Compute gradient of loss w.r.t. 3D means due to 2D mean gradients
    # Following the chain rule through the perspective projection
    mul1 = (proj_matrix[0, 0] * mean3D[0] + proj_matrix[1, 0] * mean3D[1] + 
           proj_matrix[2, 0] * mean3D[2] + proj_matrix[3, 0]) * m_w * m_w
    
    mul2 = (proj_matrix[0, 1] * mean3D[0] + proj_matrix[1, 1] * mean3D[1] + 
           proj_matrix[2, 1] * mean3D[2] + proj_matrix[3, 1]) * m_w * m_w
    
    dL_dmean = wp.vec3(0.0, 0.0, 0.0)
    
    # x component of gradient
    dL_dmean[0] = (proj_matrix[0, 0] * m_w - proj_matrix[0, 3] * mul1) * dL_dmean2D_val[0] + \
                 (proj_matrix[0, 1] * m_w - proj_matrix[0, 3] * mul2) * dL_dmean2D_val[1]
    
    # y component of gradient
    dL_dmean[1] = (proj_matrix[1, 0] * m_w - proj_matrix[1, 3] * mul1) * dL_dmean2D_val[0] + \
                 (proj_matrix[1, 1] * m_w - proj_matrix[1, 3] * mul2) * dL_dmean2D_val[1]
    
    # z component of gradient
    dL_dmean[2] = (proj_matrix[2, 0] * m_w - proj_matrix[2, 3] * mul1) * dL_dmean2D_val[0] + \
                 (proj_matrix[2, 1] * m_w - proj_matrix[2, 3] * mul2) * dL_dmean2D_val[1]
    

    dL_dmeans[idx] += dL_dmean

def backward_preprocess(
    # Camera and model parameters
    num_points: int,
    means: wp.array(dtype=wp.vec3),           # 3D means
    means_2d: wp.array(dtype=wp.vec2),        # 2D means
    radii: wp.array(dtype=int),               # Computed radii
    sh_coeffs: wp.array(dtype=wp.vec3),       # SH coefficients 
    scales: wp.array(dtype=wp.vec3),          # Scale parameters
    rotations: wp.array(dtype=wp.vec4),       # Rotation quaternions
    viewmatrix: wp.mat44,                     # Camera view matrix
    projmatrix: wp.mat44,                     # Camera projection matrix
    fov_x: float,                             # Camera horizontal FOV
    fov_y: float,                             # Camera vertical FOV
    focal_x: float,
    focal_y: float,
    
    # Intermediate data from forward
    cov3Ds: wp.array(dtype=wp.mat33),         # 3D covariance matrices (or VEC6 depending on packing)
    conic_opacity: wp.array(dtype=wp.vec4),  # 2D conics and opacity
    campos: wp.array(dtype=wp.vec3),        # View directions (should be campos)
    clamped: wp.array(dtype=wp.uint32),      # Clamping states
    
    # Incoming gradients from render backward
    dL_dmean2D: wp.array(dtype=wp.vec3),     # Grad of loss w.r.t. 2D means
    dL_dconic: wp.array(dtype=wp.vec4),      # Grad of loss w.r.t. 2D conics
    dL_dopacity: wp.array(dtype=float),      # Grad of loss w.r.t. opacity
    dL_dcolors: wp.array(dtype=wp.vec3),     # Grad of loss w.r.t. colors
    
    # Output gradient buffers
    dL_dmeans: wp.array(dtype=wp.vec3),      # Output grad for 3D means
    dL_dsh: wp.array(dtype=wp.vec3),         # Output grad for SH coeffs
    dL_dscales: wp.array(dtype=wp.vec3),     # Output grad for scales
    dL_drots: wp.array(dtype=wp.vec4),       # Output grad for rotations
    
    # Optional parameters
    scale_modifier: float = 1.0,
    sh_degree: int = 3
):
    """
    Orchestrates the backward pass for 3D Gaussian Splatting by coordinating several kernel calls.
    """
    # Create buffer for 3D covariance gradients
    dL_dcov3D = wp.zeros(num_points, dtype=VEC6, device=DEVICE)
    # Step 1: Compute gradients for 2D covariance (conic matrix)
    # This also computes gradients w.r.t. 3D means due to conic computation
    wp.launch(
        kernel=compute_cov2d_backward_kernel,
        dim=num_points,
        inputs=[
            num_points,            # P
            means,                 # means3D
            cov3Ds,                 # cov3Ds
            radii,                 # radii
            focal_x,               # focal_x
            focal_y,               # focal_y
            fov_x,                 # tan_fovx
            fov_y,                 # tan_fovy
            viewmatrix,            # viewmatrix
            dL_dconic,             # dL_dconic
            dL_dmeans,             # dL_dmean3D (outputs)
            dL_dcov3D              # dL_dcov3D (outputs)
        ],
        device=DEVICE
    )
    
    dL_dmeans_np = dL_dmeans.numpy()
    # Step 2: Compute gradients for 3D means due to projection
    wp.launch(
        kernel=compute_projection_backward_kernel,
        dim=num_points,
        inputs=[
            num_points,
            means,
            radii,
            projmatrix,
            dL_dmean2D,
            dL_dmeans  # Accumulate to final means gradients
        ],
        device=DEVICE
    )
    
    # Step 3: Compute gradients for SH coefficients
    wp.launch(
        kernel=sh_backward_kernel,
        dim=num_points,
        inputs=[
            num_points,
            sh_degree,
            means,
            sh_coeffs,
            radii,
            campos,
            clamped,
            dL_dcolors,
            dL_dmeans,
            dL_dsh
        ],
        
        device=DEVICE
    )
    dL_dmeans_np = dL_dmeans.numpy()
    # Step 4: Compute gradients for scales and rotations
    wp.launch(
        kernel=compute_cov3d_backward_kernel,
        dim=num_points,
        inputs=[
            num_points,
            scales,
            rotations,
            radii,
            scale_modifier,
            dL_dcov3D,
            dL_dscales,  # Output scale gradients
            dL_drots     # Output rotation gradients
        ],
        device=DEVICE
    )

    return dL_dmeans, dL_dsh, dL_dscales, dL_drots

def backward_render(
    ranges,
    point_list,
    width,
    height,
    bg_color,
    tile_grid,
    points_xy_image,
    conic_opacity,
    colors,
    depths,  # Added depth parameter
    final_Ts,
    n_contrib,
    dL_dpixels,
    dL_invdepths,  # Added depth gradient parameter
    use_invdepth,
    dL_dmean2D,
    dL_dconic2D,
    dL_dopacity,
    dL_dcolors,
    dL_dinvdepths,  # Added output depth gradient parameter
):
    """
    Orchestrates the backward rendering process by launching the backward kernel.
    
    Args:
        ranges: Range of point indices for each tile
        point_list: Sorted list of point indices
        width, height: Image dimensions
        bg_color: Background color
        points_xy_image: 2D positions of Gaussians
        conic_opacity: Conic matrices and opacities
        colors: RGB colors
        depths: Depth values for each Gaussian
        final_Ts: Final transparency values from forward pass
        n_contrib: Number of contributors per pixel
        dL_dpixels: Gradient of loss w.r.t. output pixels
        dL_invdepths: Gradient of loss w.r.t. inverse depths
        dL_dmean2D: Output gradient w.r.t. 2D mean positions
        dL_dconic2D: Output gradient w.r.t. conic matrices
        dL_dopacity: Output gradient w.r.t. opacity
        dL_dcolors: Output gradient w.r.t. colors
        dL_dinvdepths: Output gradient w.r.t. inverse depths
    """
    # Calculate tile grid dimensions
    tile_grid_x = (width + TILE_M - 1) // TILE_M
    tile_grid_y = (height + TILE_N - 1) // TILE_N
    # Launch the backward rendering kernel
    wp.launch(
        kernel=wp_render_backward_kernel,
        dim=(tile_grid_x, tile_grid_y, TILE_M, TILE_N),
        inputs=[
            ranges,
            point_list,
            width,
            height,
            bg_color,
            tile_grid,
            points_xy_image,
            conic_opacity,
            colors,
            depths,  # Added depth input
            final_Ts,
            n_contrib,
            dL_dpixels,
            dL_invdepths,  # Added depth gradient input
            use_invdepth,
            dL_dmean2D,
            dL_dconic2D,
            dL_dopacity,
            dL_dcolors,
            dL_dinvdepths  # Added depth gradient output
        ],
    )
    
def backward(
    # --- Core parameters ---
    background,
    means3D,
    dL_dpixels,
    # --- Model parameters ---
    opacity=None,
    shs=None,
    scales=None,
    rotations=None,
    scale_modifier=1.0,
    # --- Camera parameters ---
    viewmatrix=None,
    projmatrix=None,
    tan_fovx=0.5, 
    tan_fovy=0.5,
    image_height=256,
    image_width=256,
    campos=None,
    # --- Forward output buffers ---
    radii=None,
    means2D=None,  
    conic_opacity=None,
    rgb=None,
    clamped=None,
    depth=None,  # Added depth parameter
    cov3Ds=None,
    dL_invdepths=None,
    # --- Internal state buffers ---
    geom_buffer=None,
    binning_buffer=None,
    img_buffer=None,
    # --- Algorithm parameters ---
    degree=3,
    debug=False,
):
    """
    Main backward function for 3D Gaussian Splatting.
    
    This function orchestrates the entire backward pass by calling two main sub-functions:
    1. backward_render: Computes gradients w.r.t. 2D parameters (mean2D, conic, opacity, color)
    2. backward_preprocess: Computes gradients w.r.t. 3D parameters 
       (mean3D, cov3D, SH coefficients, scales, rotations)
    
    Args:
        background: Background color as numpy array, torch tensor, or wp.vec3 (3,)
        means3D: 3D positions as numpy array, torch tensor, or wp.array (N, 3)
        dL_dpixels: Gradient of loss w.r.t. output pixels (H, W, 3)
        opacity: Opacity values (N, 1) or (N,)
        shs: Spherical harmonics coefficients (N, D, 3) or flattened (N*D, 3)
        scales: Scale parameters (N, 3)
        rotations: Rotation quaternions (N, 4)
        scale_modifier: Global scale modifier (float)
        viewmatrix: View matrix (4, 4)
        projmatrix: Projection matrix (4, 4)
        tan_fovx: Tangent of x field of view
        tan_fovy: Tangent of y field of view
        image_height: Image height
        image_width: Image width
        campos: Camera position (3,)
        radii: Computed radii from forward pass (N,)
        means2D: 2D projected positions from forward pass (N, 2)
        conic_opacity: Conic matrices + opacity from forward pass (N, 4)
        rgb: RGB colors from forward pass (N, 3)
        clamped: Clamping state from forward pass (N, 3)
        depth: Depth values from forward pass (N,)
        cov3Ds: 3D covariance matrices from forward pass (N, 6)
        dL_invdepths: Gradient of loss w.r.t. inverse depths (N,)
        geom_buffer: Dictionary holding geometric state
        binning_buffer: Dictionary holding binning state
        img_buffer: Dictionary holding image state
        degree: SH degree (0-3)
        debug: Enable debug output
        
    Returns:
        dict: Dictionary containing gradients for all model parameters:
            - dL_dmean3D: Gradient w.r.t. 3D positions (N, 3)
            - dL_dcolor: Gradient w.r.t. colors (N, 3)
            - dL_dshs: Gradient w.r.t. SH coefficients (N*D, 3)
            - dL_dopacity: Gradient w.r.t. opacity (N,)
            - dL_dscale: Gradient w.r.t. scales (N, 3)
            - dL_drot: Gradient w.r.t. rotations (N, 4)
            - dL_dinvdepths: Gradient w.r.t. inverse depths (N,)
    """
    # Calculate focal lengths from FoV
    focal_y = image_height / (2.0 * tan_fovy)
    focal_x = image_width / (2.0 * tan_fovx)

    # Convert inputs to warp arrays
    background_warp = background if isinstance(background, wp.vec3) else wp.vec3(background[0], background[1], background[2])
    means3D_warp = to_warp_array(means3D, wp.vec3)
    dL_dpixels_warp = to_warp_array(dL_dpixels, wp.vec3) if not isinstance(dL_dpixels, wp.array) else dL_dpixels
    
    # Get number of points
    num_points = means3D_warp.shape[0]
    
    # Convert optional parameters if provided
    opacity_warp = to_warp_array(opacity, float, flatten=True) if opacity is not None else None
    
    # SH coefficients need special handling for flattening
    if shs is not None:
        sh_data = shs.reshape(-1, 3) if hasattr(shs, 'reshape') and shs.ndim > 2 else shs
        shs_warp = to_warp_array(sh_data, wp.vec3)
    else:
        shs_warp = None
    
    # Handle other model parameters
    scales_warp = to_warp_array(scales, wp.vec3) if scales is not None else None
    
    # Handle rotations differently based on shape (matrices vs quaternions)
    if rotations is not None:
        rot_shape = rotations.shape[-1] if hasattr(rotations, 'shape') else rotations.size(-1)
        if rot_shape == 4:  # Quaternions
            rotations_warp = to_warp_array(rotations, wp.vec4)
        else:  # 3x3 matrices
            rotations_warp = to_warp_array(rotations, wp.mat33)
    else:
        rotations_warp = None
    
    # Handle camera parameters
    viewmatrix_warp = viewmatrix if isinstance(viewmatrix, wp.mat44) else wp.mat44(viewmatrix.flatten())
    projmatrix_warp = projmatrix if isinstance(projmatrix, wp.mat44) else wp.mat44(projmatrix.flatten())
    campos_warp = campos if isinstance(campos, wp.vec3) else wp.vec3(campos[0], campos[1], campos[2])
    
    # --- Extract data from buffer dictionaries if provided ---
    if img_buffer is not None:
        ranges = img_buffer.get('ranges')
        final_Ts = img_buffer.get('final_Ts')
        n_contrib = img_buffer.get('n_contrib')
    
    if binning_buffer is not None:
        point_list = binning_buffer.get('point_list')
    
    if geom_buffer is not None:
        # Use internal data if not provided directly
        if radii is None:
            radii = geom_buffer.get('radii')
        if means2D is None:
            means2D = geom_buffer.get('means2D')
        if conic_opacity is None:
            conic_opacity = geom_buffer.get('conic_opacity')
        if rgb is None:
            rgb = geom_buffer.get('rgb')
        if clamped is None:
            clamped = geom_buffer.get('clamped_state')
        if depth is None:
            depth = geom_buffer.get('depth')  # Get depth from geom buffer if available
    
    use_invdepth = False if dL_invdepths is None else True
    if use_invdepth:
        print("WARNING: Using inverse depth gradients is ")

    # Convert forward pass outputs to warp arrays if they're not already
    radii_warp = to_warp_array(radii, int) if radii is not None else None
    means2D_warp = to_warp_array(means2D, wp.vec2) if means2D is not None else None
    conic_opacity_warp = to_warp_array(conic_opacity, wp.vec4) if conic_opacity is not None else None
    rgb_warp = to_warp_array(rgb, wp.vec3) if rgb is not None else None
    clamped_warp = to_warp_array(clamped, wp.uint32) if clamped is not None else None
    depth_warp = to_warp_array(depth, float) if depth is not None else None
    dL_invdepths_warp = to_warp_array(dL_invdepths, float) if dL_invdepths is not None else None
    
    # --- Initialize output gradient arrays ---
    dL_dmean2D = wp.zeros(num_points, dtype=wp.vec3, device=DEVICE)
    dL_dconic = wp.zeros(num_points, dtype=wp.vec4, device=DEVICE)
    dL_dopacity = wp.zeros(num_points, dtype=float, device=DEVICE)
    dL_dcolor = wp.zeros(num_points, dtype=wp.vec3, device=DEVICE)
    dL_dinvdepths = wp.zeros(num_points, dtype=float, device=DEVICE)  # Initialize depth gradient output
    
    dL_dmean3D = wp.zeros(num_points, dtype=wp.vec3, device=DEVICE)
    dL_dcov3D = wp.zeros(num_points, dtype=VEC6, device=DEVICE)
    
    # SH gradients depend on degree
    max_sh_coeffs = 16 if degree >= 3 else (degree + 1) * (degree + 1)
    dL_dsh = wp.zeros(num_points * max_sh_coeffs, dtype=wp.vec3, device=DEVICE)
    
    dL_dscale = wp.zeros(num_points, dtype=wp.vec3, device=DEVICE)
    dL_drot = wp.zeros(num_points, dtype=wp.vec4, device=DEVICE)
    
    # Use precomputed colors if provided, otherwise use colors from forward pass
    
    tile_grid = wp.vec3((image_width + TILE_M - 1) // TILE_M, 
                        (image_height + TILE_N - 1) // TILE_N, 
                        1)

    # --- Step 1: Compute loss gradients w.r.t. 2D parameters ---
    backward_render(
        ranges=ranges,
        point_list=point_list,
        width=image_width,
        height=image_height,
        bg_color=background_warp,
        tile_grid=tile_grid,
        points_xy_image=means2D_warp,
        conic_opacity=conic_opacity_warp,
        colors=rgb_warp,
        depths=depth_warp,  # Pass depth values
        final_Ts=final_Ts,
        n_contrib=n_contrib,
        dL_dpixels=dL_dpixels_warp,
        dL_invdepths=dL_invdepths_warp,  # Pass depth gradients
        use_invdepth=use_invdepth,
        dL_dmean2D=dL_dmean2D,
        dL_dconic2D=dL_dconic,
        dL_dopacity=dL_dopacity,
        dL_dcolors=dL_dcolor,
        dL_dinvdepths=dL_dinvdepths  # Pass depth gradient output
    )
    # --- Step 2: Compute gradients for 3D parameters ---
    backward_preprocess(
        num_points=num_points,
        means=means3D_warp,
        means_2d=means2D_warp,
        radii=radii_warp,
        sh_coeffs=shs_warp,
        scales=scales_warp,
        rotations=rotations_warp,
        viewmatrix=viewmatrix_warp,
        projmatrix=projmatrix_warp,
        fov_x=tan_fovx,
        fov_y=tan_fovy,
        focal_x=focal_x,
        focal_y=focal_y,
        cov3Ds=cov3Ds,
        conic_opacity=conic_opacity_warp,
        campos=campos_warp,
        clamped=clamped_warp,
        dL_dmean2D=dL_dmean2D,
        dL_dconic=dL_dconic,
        dL_dopacity=dL_dopacity,
        dL_dcolors=dL_dcolor,
        dL_dmeans=dL_dmean3D,
        dL_dsh=dL_dsh,
        dL_dscales=dL_dscale,
        dL_drots=dL_drot,
        sh_degree=degree
    )
    
    # Return all gradients in a dictionary for easy access
    return {
        'dL_dmean3D': dL_dmean3D,
        'dL_dcolor': dL_dcolor,
        'dL_dshs': dL_dsh,
        'dL_dopacity': dL_dopacity,
        'dL_dscale': dL_dscale,
        'dL_drot': dL_drot,
        'dL_dinvdepths': dL_dinvdepths,  # Include depth gradients in return
        # Include 2D gradients for completeness
        'dL_dmean2D': dL_dmean2D,
        'dL_dconic': dL_dconic,
        'dL_dcov3D': dL_dcov3D
    }

@wp.kernel
def densify_gaussians(
    positions: wp.array(dtype=wp.vec3),
    pos_grads: wp.array(dtype=wp.vec3),
    scale_grads: wp.array(dtype=wp.vec3),
    num_points: int,
    noise_scale: float
):
    i = wp.tid()
    if i >= num_points:
        return
    
    # Get gradient magnitudes to identify high-gradient Gaussians
    pos_grad_mag = wp.length(pos_grads[i])
    scale_grad_mag = wp.length(scale_grads[i])
    
    # We just add noise to positions based on gradients
    # In a full implementation, you'd clone Gaussians with high gradient magnitude
    if pos_grad_mag > 0.1 or scale_grad_mag > 0.1:
        # Add random noise to position
        seed = wp.uint32(i)
        noise = wp.vec3(
            wp.randn(seed) * noise_scale,
            wp.randn(seed + wp.uint32(1000)) * noise_scale,
            wp.randn(seed + wp.uint32(2000)) * noise_scale
        )
        positions[i] = positions[i] + noise

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


@wp.func
def wp_vec3_clamp(x: wp.vec3, min_val: float, max_val: float) -> wp.vec3:
    return wp.vec3(
        wp.clamp(x[0], min_val, max_val),
        wp.clamp(x[1], min_val, max_val),
        wp.clamp(x[2], min_val, max_val)
    )

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
    
    # Clamp gradients to avoid exploding gradients
    pos_grads[i] = wp_vec3_clamp(pos_grads[i], -1e3, 1e3)
    scale_grads[i] = wp_vec3_clamp(scale_grads[i], -1e3, 1e3)
    rot_grads[i] = wp.vec4(
        wp.clamp(rot_grads[i][0], -1e3, 1e3),
        wp.clamp(rot_grads[i][1], -1e3, 1e3),
        wp.clamp(rot_grads[i][2], -1e3, 1e3),
        wp.clamp(rot_grads[i][3], -1e3, 1e3)
    )
    opacity_grads[i] = wp.clamp(opacity_grads[i], -100.0, 100.0)
    
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

    
    opacities[i] = wp.clamp(opacities[i], 1e-4, 0.20)

    # 3) scales 更新完立刻裁剪
    scales[i] = wp.vec3(
        wp.clamp(scales[i][0], 1e-3, 0.30),
        wp.clamp(scales[i][1], 1e-3, 0.30),
        wp.clamp(scales[i][2], 1e-3, 0.30),
    )
