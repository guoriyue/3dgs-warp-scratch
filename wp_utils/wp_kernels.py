import numpy as np
import warp as wp

# half precision
WP_FLOAT16 = wp.float16
# originally using bf16 but warp only supports float16 for now
WP_FLOAT32 = wp.float32

WP_INT = wp.int32
WP_VEC2 = wp.vec2
WP_VEC2H = wp.vec2h

DEVICE = "cuda:0"  # "cpu"

# Initialize Warp
wp.init()

syncthreads_snippet = """
__syncthreads();
"""

@wp.func_native(syncthreads_snippet)
def syncthreads():
    return

@wp.kernel
def wp_embedding(embedding: wp.array2d(dtype=WP_FLOAT32), tokens: wp.array(dtype=WP_INT), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    idx = tokens[i]
    output[i, j] = embedding[idx, j]

@wp.kernel
def wp_pow(tensor: wp.array2d(dtype=WP_FLOAT32), power: WP_FLOAT32, output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j] = wp.pow(tensor[i, j], power)

@wp.kernel
def wp_mean(tensor: wp.array2d(dtype=WP_FLOAT32), axis: WP_INT, output: wp.array(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    if axis == 0:
        wp.atomic_add(output, i, tensor[i, j])
    elif axis == 1:
        wp.atomic_add(output, j, tensor[i, j])
    
@wp.kernel
def wp_sqrt(tensor: wp.array2d(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j] = wp.sqrt(tensor[i, j])

@wp.kernel
def wp_rms_norm(tensor: wp.array2d(dtype=WP_FLOAT32), norm_eps:WP_FLOAT32, norm_weights: wp.array(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i = wp.tid()
    tensor_2_mean_tid = WP_FLOAT32(0.0)
    for j in range(tensor.shape[1]):
        tensor_2_mean_tid = tensor_2_mean_tid + tensor[i, j] * tensor[i, j]
    tensor_2_mean_tid = tensor_2_mean_tid / WP_FLOAT32(tensor.shape[1]) + norm_eps
    rms_tid = wp.sqrt(tensor_2_mean_tid)
    for j in range(tensor.shape[1]):
        output[i, j] = tensor[i, j] * norm_weights[j] / rms_tid
        
@wp.kernel
def wp_compute_freqs(rope_theta: WP_FLOAT32, zero_to_one_split_into_64_parts: wp.array(dtype=WP_FLOAT32), freqs: wp.array(dtype=WP_FLOAT32)):
    tid = wp.tid()
    freqs[tid] = WP_FLOAT32(1.0) / wp.pow(rope_theta, zero_to_one_split_into_64_parts[tid])
    
@wp.kernel
def wp_outer_1d_1d_2d(a: wp.array(dtype=WP_FLOAT32), b: wp.array(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j] = a[i] * b[j]
        
@wp.kernel
def wp_polar(inputs: wp.array2d(dtype=WP_FLOAT32), outputs: wp.array3d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    outputs[i, j][0] = wp.cos(inputs[i, j])
    outputs[i, j][1] = wp.sin(inputs[i, j])

@wp.kernel
def wp_complex_multiply(complex1: wp.array3d(dtype=WP_FLOAT32), complex2: wp.array3d(dtype=WP_FLOAT32), output: wp.array3d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    # Extract real and imaginary parts
    real1 = complex1[i, j][0]
    imag1 = complex1[i, j][1]
    real2 = complex2[i, j][0]
    imag2 = complex2[i, j][1]

    # Perform complex multiplication
    real_output = real1 * real2 - imag1 * imag2
    imag_output = real1 * imag2 + imag1 * real2

    # Store the result in the output array
    output[i, j][0] = real_output
    output[i, j][1] = imag_output

@wp.kernel
def wp_triu(mask: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()  # Get the thread indices
    if j < i + 1:
        mask[i, j] = 0.0  # Keep the values above the diagonal (including diagonal) as 0

@wp.kernel
def wp_softmax_exp_sum(qk: wp.array2d(dtype=WP_FLOAT32), mask: wp.array2d(dtype=WP_FLOAT32), exp: wp.array(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    wp.atomic_add(exp, i, wp.exp(qk[i, j] + mask[i, j]))

@wp.kernel
def wp_softmax_exp_normalize(qk: wp.array2d(dtype=WP_FLOAT32), mask: wp.array2d(dtype=WP_FLOAT32), exp: wp.array(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j] = wp.exp(qk[i, j] + mask[i, j]) / exp[i]
    
@wp.kernel
def wp_add_2d(a: wp.array2d(dtype=WP_FLOAT32), b: wp.array2d(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j] = a[i, j] + b[i, j]

@wp.kernel
def wp_add_1d(a: wp.array(dtype=WP_FLOAT32), b: wp.array(dtype=WP_FLOAT32), output: wp.array(dtype=WP_FLOAT32)):
    i = wp.tid()
    output[i] = a[i] + b[i]

@wp.kernel
def wp_add_1d_scalar_int32(tensor: wp.array(dtype=WP_INT), scalar: WP_INT, output: wp.array(dtype=WP_INT)):
    i = wp.tid()
    output[i] = tensor[i] + scalar

@wp.kernel
def wp_mul_2d(a: wp.array2d(dtype=WP_FLOAT32), b: wp.array2d(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j] = a[i, j] * b[i, j]

@wp.kernel
def wp_mul_1d_scalar(a: wp.array(dtype=WP_FLOAT32), scalar: WP_FLOAT32, output: wp.array(dtype=WP_FLOAT32)):
    i = wp.tid()
    output[i] = a[i] * scalar
    
@wp.kernel
def wp_stack(head: WP_INT, stack_tensor: wp.array2d(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j + head * stack_tensor.shape[1]] = stack_tensor[i, j]

@wp.kernel
def wp_silu(x: wp.array2d(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()  # Get the row index
    sig = 1.0 / (1.0 + wp.exp(-x[i, j]))  # Sigmoid function
    output[i, j] = x[i, j] * sig  # SiLU activation
    
@wp.kernel
def wp_access_d1(inputs: wp.array2d(dtype=WP_FLOAT32), outputs: wp.array2d(dtype=WP_FLOAT32), index: WP_INT):
    j = wp.tid()
    outputs[0, j] = inputs[index, j]
    
@wp.kernel
def wp_argmax(logits: wp.array2d(dtype=WP_FLOAT32), max_logit: wp.array(dtype=WP_FLOAT32)):
    i = wp.tid()
    wp.atomic_max(max_logit, 0, logits[0, i])

@wp.kernel
def wp_index(logits: wp.array2d(dtype=WP_FLOAT32), logit_value: wp.array(dtype=WP_FLOAT32), logit_index: wp.array(dtype=WP_INT)):
    i = wp.tid()
    if logits[0, i] == logit_value[0]:
        logit_index[0] = i

@wp.kernel
def wp_arange(output: wp.array(dtype=WP_INT)):
    i = wp.tid()
    output[i] = i

@wp.kernel
def wp_meshgrid(x: wp.array(dtype=WP_FLOAT32), y: wp.array(dtype=WP_FLOAT32), xx: wp.array2d(dtype=WP_FLOAT32), yy: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    xx[i, j] = x[j]
    yy[i, j] = y[i]