from wp_utils.wp_kernels import *

def wp_arange_py(n):
    output = wp.zeros(n, dtype=WP_INT)
    wp.launch(kernel=wp_arange, dim=(n,), inputs=[output])
    return output


def wp_add_1d_scalar_int32_py(a, b):
    output = wp.zeros(a.shape, dtype=WP_INT)
    wp_b = wp.int32(b)
    wp.launch(kernel=wp_add_1d_scalar_int32, dim=a.shape, inputs=[a, wp_b, output])
    return output

def wp_meshgrid_py(x, y):
    xx = wp.zeros((x.shape[0], y.shape[0]), dtype=WP_FLOAT32)
    yy = wp.zeros((x.shape[0], y.shape[0]), dtype=WP_FLOAT32)
    wp.launch(kernel=wp_meshgrid, dim=(x.shape[0], y.shape[0]), inputs=[x, y, xx, yy])
    return xx, yy

def wp_mul_scalar_py(a, b):
    output = wp.zeros(a.shape, dtype=WP_FLOAT32)
    wp_b = wp.float32(b)
    wp.launch(kernel=wp_mul_1d_scalar, dim=a.shape, inputs=[a, wp_b, output])
    return output

def wp_stack_py(to_stack):
    head = len(to_stack)
    stack_tensor = wp.array2d(to_stack[0], dtype=WP_FLOAT32)
    output = wp.zeros((stack_tensor.shape[0], stack_tensor.shape[1]*head), dtype=WP_FLOAT32)
    wp.launch(kernel=wp_stack, dim=(stack_tensor.shape[0], stack_tensor.shape[1]), inputs=[head, stack_tensor, output])
    return output