import warp as wp
import numpy as np

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

def wp_arange(n):
    dtype = WP_INT
    output = wp.zeros(n, dtype=dtype)
    @wp.kernel
    def kernel_func(output: wp.array(dtype=dtype)):
        i = wp.tid()
        output[i] = i
    wp.launch(kernel_func, dim=(n,), inputs=[output])
    return output

def wp_add_1d_scalar(a, b):
    dtype = a.dtype
    output = wp.zeros(a.shape, dtype=dtype)
    wp_b = wp.int32(b)
    @wp.kernel
    def kernel_func(a: wp.array(dtype=dtype), b: wp.int32, output: wp.array(dtype=dtype)):
        i = wp.tid()
        output[i] = a[i] + b
    wp.launch(kernel=kernel_func, dim=a.shape, inputs=[a, wp_b, output])
    return output

def wp_meshgrid(x, y):
    dtype = x.dtype
    xx = wp.zeros((x.shape[0], y.shape[0]), dtype=dtype)
    yy = wp.zeros((x.shape[0], y.shape[0]), dtype=dtype)

    @wp.kernel
    def kernel_func(x: wp.array(dtype=dtype), y: wp.array(dtype=dtype), xx: wp.array2d(dtype=dtype), yy: wp.array2d(dtype=dtype)):
        i, j = wp.tid()
        xx[i, j] = x[i]
        yy[i, j] = y[j]
    
    wp.launch(kernel=kernel_func, dim=(x.shape[0], y.shape[0]), inputs=[x, y, xx, yy])
    return xx, yy

def wp_to_float32_2d(a):
    output = wp.zeros(a.shape, dtype=WP_FLOAT32)

    @wp.kernel
    def kernel_func(a: wp.array2d(dtype=a.dtype), output: wp.array2d(dtype=WP_FLOAT32)):
        i, j = wp.tid()
        output[i, j] = wp.float32(a[i, j])
    
    wp.launch(kernel=kernel_func, dim=a.shape, inputs=[a, output])
    return output

def wp_dot_prod_2d_scalar(a: wp.array2d, scalar):
    dtype = a.dtype
    output = wp.zeros(a.shape, dtype=dtype)

    @wp.kernel
    def kernel_func(a: wp.array2d(dtype=dtype), scalar: dtype, output: wp.array2d(dtype=dtype)):
        i, j = wp.tid()
        output[i, j] = a[i, j] * scalar

    wp.launch(kernel_func, dim=a.shape, inputs=[a, scalar, output])
    return output

def wp_stack_2d(to_stack):
    head = len(to_stack)
    if head == 0:
        print("Nothing to stack")
        return None
    
    dtype = to_stack[0].dtype
    output = wp.zeros((head, to_stack[0].shape[0], to_stack[0].shape[1]), dtype=dtype)
    
    @wp.kernel
    def kernel_func(head: WP_INT, stack_tensor: wp.array2d(dtype=dtype), output: wp.array3d(dtype=dtype)):
        i, j = wp.tid()
        output[head, i, j] = stack_tensor[i, j]

    for h in range(head):
        wp.launch(kernel_func, dim=(to_stack[h].shape[0], to_stack[h].shape[1]), inputs=[h, to_stack[h], output])
    
    return output



def wp_permute_3d(a, perm):
    dtype = a.dtype
    perm_shape = [a.shape[i] for i in perm]
    numel = int(np.prod(perm_shape))
    output = wp.zeros(perm_shape, dtype=dtype)

    perm = wp.array(perm, dtype=WP_INT)

    @wp.kernel
    def kernel_func(input_tensor: wp.array3d(dtype=float), output_tensor: wp.array3d(dtype=float)):
        i = wp.tid()  # Global thread index
        d1, d2, d3 = input_tensor.shape

        d1_idx = i // (d2 * d3)
        d23_idx = i % (d2 * d3)
        d2_idx = d23_idx // d3
        d3_idx = d23_idx % d3

        output_tensor[d1_idx, d3_idx, d2_idx] = input_tensor[d1_idx, d2_idx, d3_idx]
        

    wp.launch(kernel_func, dim=numel, inputs=[a, output])

    return output