import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import math
from wp_kernels import render_gaussians, wp_world_to_view, wp_projection_matrix
import imageio
from config import *

# Initialize Warp
wp.init()

if __name__ == "__main__":
    pts = np.array([[2, 0, -2], [0, 2, -2], [-2, 0, -2]], dtype=np.float32)
    n = len(pts)
    shs = np.random.random((n, 16, 3)).astype(np.float32)
    opacities = np.ones((n, 1), dtype=np.float32)  # Match 3dgs.py format exactly

    scales = np.ones((n, 3), dtype=np.float32)
    rotations = np.array([np.eye(3)] * n, dtype=np.float32)

    # Set camera parameters exactly like in 3dgs.py
    camera_pos = np.array([0, 0, 5], dtype=np.float32)
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)

    # Image parameters
    image_width = 700
    image_height = 700
    aspect_ratio = image_width / image_height
    fovx = 45.0
    fovy = 45.0
    znear = 0.01
    zfar = 100.0

    # Background color (black as in 3dgs.py)
    background = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Scale modifier
    scale_modifier = 1.0
    
    
    view_matrix = wp_world_to_view(R=R, t=camera_pos)
    proj_matrix = wp_projection_matrix(fovx=fovx, fovy=fovy, znear=znear, zfar=zfar)
    proj_matrix = np.dot(proj_matrix, view_matrix)
    
    
    tan_fovx = math.tan(fovx * 0.5)
    tan_fovy = math.tan(fovy * 0.5)

    # Call the Gaussian rasterizer
    rendered_image, depth_image = render_gaussians(
        background=background,
        pts=pts,
        colors=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        scale_modifier=scale_modifier,
        view_matrix=view_matrix,
        proj_matrix=proj_matrix,
        tan_fovx=tan_fovx,
        tan_fovy=tan_fovy,
        sh_coeffs=shs,
        image_height=image_height,
        image_width=image_width,
        camera_pos=camera_pos,
        antialiasing=False,
        verbose=True
    )

    # Convert the rendered image from device to host
    rendered_array = wp.to_torch(rendered_image).cpu().numpy()

    # Filter out low values and ensure colors are in valid range
    rendered_array = np.where(rendered_array < 0.1, 0.0, rendered_array)
    rendered_array = np.clip(rendered_array, 0.0, 1.0)

    # Display and save using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(rendered_array)
    plt.axis('off')
    plt.savefig("gaussian_render.png", bbox_inches='tight', dpi=150)
