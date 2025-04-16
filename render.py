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
    # shs = np.random.random((n, 16, 3))
    # randomly generate shs, hard code here for debugging
    shs = np.array([[0.71734341, 0.91905449, 0.49961076],
                [0.08068483, 0.82132256, 0.01301602],
                [0.8335743,  0.31798138, 0.19709007],
                [0.82589597, 0.28206231, 0.790489  ],
                [0.24008527, 0.21312673, 0.53132892],
                [0.19493135, 0.37989934, 0.61886235],
                [0.98106522, 0.28960672, 0.57313965],
                [0.92623716, 0.46034381, 0.5485369 ],
                [0.81660616, 0.7801104,  0.27813915],
                [0.96114063, 0.69872817, 0.68313804],
                [0.95464185, 0.21984855, 0.92912192],
                [0.23503135, 0.29786121, 0.24999751],
                [0.29844887, 0.6327788,  0.05423596],
                [0.08934335, 0.11851827, 0.04186001],
                [0.59331831, 0.919777,   0.71364335],
                [0.83377388, 0.40242542, 0.8792624 ]]*n).reshape(n, 16, 3) 
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
    
    focal_x = image_width / (2 * tan_fovx)
    focal_y = image_height / (2 * tan_fovy)
    
    colors = np.random.random((n, 3)).astype(np.float32)

    # Call the Gaussian rasterizer
    rendered_image, depth_image = render_gaussians(
        background=background,
        means3D=pts,
        colors=colors,
        opacity=opacities,
        scales=scales,
        rotations=rotations,
        scale_modifier=scale_modifier,
        viewmatrix=view_matrix,
        projmatrix=proj_matrix,
        tan_fovx=tan_fovx,
        tan_fovy=tan_fovy,
        image_height=image_height,
        image_width=image_width,
        sh=shs,
        degree=3,
        campos=camera_pos,
        prefiltered=False,
        antialiasing=False,
        clamped=True,
        debug=True
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
