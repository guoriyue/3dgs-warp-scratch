import warp as wp
import numpy as np
from config import DEVICE
from utils import *

# Constants for SSIM calculation
C1 = 0.01 ** 2
C2 = 0.03 ** 2
WINDOW_SIZE = 11

@wp.kernel
def l1_loss_kernel(
    rendered: wp.array2d(dtype=wp.vec3),
    target: wp.array2d(dtype=wp.vec3),
    loss_buffer: wp.array(dtype=float),
    width: int,
    height: int
):
    i, j = wp.tid()
    if i >= width or j >= height:
        return
    
    # Compute L1 difference for each pixel component
    rendered_pixel = rendered[j, i]
    target_pixel = target[j, i]
    diff = wp.abs(rendered_pixel - target_pixel)
    l1_diff = diff[0] + diff[1] + diff[2]
    
    # Atomic add to loss buffer
    wp.atomic_add(loss_buffer, 0, l1_diff)


@wp.kernel
def gaussian_kernel(
    kernel: wp.array(dtype=float),
    sigma: float,
    kernel_size: int
):
    i = wp.tid()
    if i >= kernel_size:
        return
    
    center = kernel_size // 2
    x = i - center
    kernel[i] = wp.exp(-1.0 * float(x * x) / (2.0 * sigma * sigma))

@wp.kernel
def ssim_kernel(
    rendered: wp.array2d(dtype=wp.vec3),
    target: wp.array2d(dtype=wp.vec3),
    gaussian_weights: wp.array(dtype=float),
    ssim_buffer: wp.array(dtype=float),
    width: int,
    height: int,
    window_size: int
):
    i, j = wp.tid()
    if i >= width or j >= height:
        return
    
    # Constants for numerical stability
    c1 = 0.01 * 0.01
    c2 = 0.03 * 0.03
    
    # We'll compute SSIM in a local window around each pixel
    half_window = window_size // 2
    
    # Initialize accumulators
    mu1 = wp.vec3(0.0, 0.0, 0.0)
    mu2 = wp.vec3(0.0, 0.0, 0.0)
    sigma1 = wp.vec3(0.0, 0.0, 0.0)
    sigma2 = wp.vec3(0.0, 0.0, 0.0)
    sigma12 = wp.vec3(0.0, 0.0, 0.0)
    weight_sum = float(0.0)
    
    # Calculate weighted means and variances over the window
    for y in range(max(0, j - half_window), min(height, j + half_window + 1)):
        for x in range(max(0, i - half_window), min(width, i + half_window + 1)):
            # Get Gaussian weight for this position
            wy = abs(y - j)
            wx = abs(x - i)
            if wx <= half_window and wy <= half_window:
                w = gaussian_weights[wx] * gaussian_weights[wy]
                
                # Get pixels
                p1 = rendered[y, x]
                p2 = target[y, x]
                
                # Accumulate weighted values
                mu1 += p1 * w
                mu2 += p2 * w
                sigma1 += wp_vec3_mul_element(p1, p1) * w
                sigma2 += wp_vec3_mul_element(p2, p2) * w
                sigma12 += wp_vec3_mul_element(p1, p2) * w
                weight_sum += w
    
    # Normalize by weights
    if weight_sum > 0.0:
        mu1 /= weight_sum
        mu2 /= weight_sum
        sigma1 /= weight_sum
        sigma2 /= weight_sum
        sigma12 /= weight_sum
    
    # Calculate variance and covariance
    sigma1 = sigma1 - wp_vec3_mul_element(mu1, mu1)
    sigma2 = sigma2 - wp_vec3_mul_element(mu2, mu2)
    sigma12 = sigma12 - wp_vec3_mul_element(mu1, mu2)
    
    # Calculate SSIM for each channel
    ssim_r = ((2.0 * mu1[0] * mu2[0] + c1) * (2.0 * sigma12[0] + c2)) / ((mu1[0] * mu1[0] + mu2[0] * mu2[0] + c1) * (sigma1[0] + sigma2[0] + c2))
    ssim_g = ((2.0 * mu1[1] * mu2[1] + c1) * (2.0 * sigma12[1] + c2)) / ((mu1[1] * mu1[1] + mu2[1] * mu2[1] + c1) * (sigma1[1] + sigma2[1] + c2))
    ssim_b = ((2.0 * mu1[2] * mu2[2] + c1) * (2.0 * sigma12[2] + c2)) / ((mu1[2] * mu1[2] + mu2[2] * mu2[2] + c1) * (sigma1[2] + sigma2[2] + c2))
    
    # Average SSIM across channels
    ssim_val = (ssim_r + ssim_g + ssim_b) / 3.0
    
    # Atomic add to SSIM buffer
    wp.atomic_add(ssim_buffer, 0, ssim_val)

@wp.kernel
def backprop_l1_pixel_gradients(
    rendered: wp.array2d(dtype=wp.vec3),
    target: wp.array2d(dtype=wp.vec3),
    pixel_grad: wp.array2d(dtype=wp.vec3),
    width: int,
    height: int,
    l1_weight: float
):
    i, j = wp.tid()
    if i >= width or j >= height:
        return
    
    # Compute gradient (sign function for L1 loss)
    rendered_pixel = rendered[j, i]
    target_pixel = target[j, i]
    
    # Sign function for L1 gradient
    l1_grad = wp.vec3(
        l1_weight * wp.sign(rendered_pixel[0] - target_pixel[0]),
        l1_weight * wp.sign(rendered_pixel[1] - target_pixel[1]),
        l1_weight * wp.sign(rendered_pixel[2] - target_pixel[2])
    )
    
    # Store L1 gradients
    pixel_grad[j, i] = l1_grad

# Python wrapper functions
def l1_loss(rendered, target):
    """Compute L1 loss between rendered and target images"""
    height, width = rendered.shape[0], rendered.shape[1]
    
    # Create device arrays if not already
    if not isinstance(rendered, wp.array):
        d_rendered = wp.array(rendered, dtype=wp.vec3, device=DEVICE)
    else:
        d_rendered = rendered
    
    if not isinstance(target, wp.array):
        d_target = wp.array(target, dtype=wp.vec3, device=DEVICE)
    else:
        d_target = target
    
    # Create loss buffer
    loss_buffer = wp.zeros(1, dtype=float, device=DEVICE)
    
    # Compute loss
    wp.launch(
        kernel=l1_loss_kernel,
        dim=(width, height),
        inputs=[d_rendered, d_target, loss_buffer, width, height]
    )
    
    # Get loss value
    loss = float(loss_buffer.numpy()[0]) / (width * height * 3)  # Normalize by pixel count and channels
    return loss

def ssim(rendered, target):
    """Compute SSIM between rendered and target images"""
    height, width = rendered.shape[0], rendered.shape[1]
    
    # Create device arrays if not already
    if not isinstance(rendered, wp.array):
        d_rendered = wp.array(rendered, dtype=wp.vec3, device=DEVICE)
    else:
        d_rendered = rendered
    
    if not isinstance(target, wp.array):
        d_target = wp.array(target, dtype=wp.vec3, device=DEVICE)
    else:
        d_target = target
    
    # Precompute Gaussian kernel
    kernel_size = WINDOW_SIZE
    gaussian_weights = wp.zeros(kernel_size, dtype=float, device=DEVICE)
    wp.launch(
        gaussian_kernel,
        dim=kernel_size,
        inputs=[gaussian_weights, 1.5, kernel_size]
    )
    
    # Create SSIM buffer
    ssim_buffer = wp.zeros(1, dtype=float, device=DEVICE)
    pixel_count = wp.zeros(1, dtype=int, device=DEVICE)
    
    # Compute SSIM
    wp.launch(
        ssim_kernel,
        dim=(width, height),
        inputs=[d_rendered, d_target, gaussian_weights, ssim_buffer, width, height, kernel_size]
    )
    
    # Get SSIM value (average over valid pixels)
    ssim_val = float(ssim_buffer.numpy()[0]) / (width * height)
    return ssim_val

def compute_image_gradients(rendered, target, lambda_dssim=0.2):
    """Compute gradients for combined L1 and SSIM loss"""
    height, width = rendered.shape[0], rendered.shape[1]
    
    # Create device arrays if not already
    if not isinstance(rendered, wp.array):
        d_rendered = wp.array(rendered, dtype=wp.vec3, device=DEVICE)
    else:
        d_rendered = rendered
    
    if not isinstance(target, wp.array):
        d_target = wp.array(target, dtype=wp.vec3, device=DEVICE)
    else:
        d_target = target
    
    # Create gradient buffer
    pixel_grad = wp.zeros((height, width), dtype=wp.vec3, device=DEVICE)
    
    # Compute L1 loss gradient
    l1_weight = 1.0 - lambda_dssim
    wp.launch(
        backprop_l1_pixel_gradients,
        dim=(width, height),
        inputs=[d_rendered, d_target, pixel_grad, width, height, l1_weight]
    )
    
    # TODO: Add SSIM gradient
    
    return pixel_grad
