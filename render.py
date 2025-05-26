import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import math
from forward import render_gaussians
from utils.math_utils import world_to_view, projection_matrix

# Initialize Warp
wp.init()

def setup_example_scene(image_width=1800, image_height=1800, fovx=45.0, fovy=45.0, znear=0.01, zfar=100.0):
    """Setup example scene with camera and Gaussians for testing and debugging"""
    # Camera setup
    T = np.array([0, 0, 5], dtype=np.float32)
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = R
    world_to_camera[:3, 3] = T
    world_to_camera = world_to_camera.T
    
    # Compute matrices
    view_matrix = world_to_view(R=R, t=T)
    proj_matrix = projection_matrix(fovx=fovx, fovy=fovy, znear=znear, zfar=zfar).T
    full_proj_matrix = world_to_camera @ proj_matrix
    
    camera_center = np.linalg.inv(world_to_camera)[3, :3]
    
    # Compute FOV parameters
    tan_fovx = math.tan(fovx * 0.5)
    tan_fovy = math.tan(fovy * 0.5)
    
    focal_x = image_width / (2 * tan_fovx)
    focal_y = image_height / (2 * tan_fovy)
    
    camera_params = {
        'R': R,
        'T': T,
        'camera_center': camera_center,
        'view_matrix': view_matrix,
        'proj_matrix': proj_matrix,
        'world_to_camera': world_to_camera,
        'full_proj_matrix': full_proj_matrix,
        'tan_fovx': tan_fovx,
        'tan_fovy': tan_fovy,
        'focal_x': focal_x,
        'focal_y': focal_y,
        'width': image_width,
        'height': image_height
    }
    
    # Gaussian setup - 3 points in a line
    pts = np.array([[-5, 0, -10], [0, 0, -10], [5, 0, -10]], dtype=np.float32)
    n = len(pts)
    
    # Hard-coded SHs for debugging
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
    
    opacities = np.ones((n, 1), dtype=np.float32)
    scales = np.ones((n, 3), dtype=np.float32)
    
    # Create quaternion rotations (identity quaternions)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 3] = 1.0  # Set w component to 1.0
    
    colors = np.ones((n, 3), dtype=np.float32)
    
    return pts, shs, scales, colors, rotations, opacities, camera_params

if __name__ == "__main__":
    # Setup rendering parameters
    image_width = 1800
    image_height = 1800
    background = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Black background
    scale_modifier = 1.0
    sh_degree = 3
    prefiltered = False
    antialiasing = False
    clamped = True
    
    # Create example scene
    pts, shs, scales, colors, rotations, opacities, camera_params = setup_example_scene(
        image_width=image_width,
        image_height=image_height
    )
    n = len(pts)
    print(f"Created example scene with {n} Gaussians")
    
    # Call the Gaussian rasterizer
    rendered_image, depth_image, _ = render_gaussians(
        background=background,
        means3D=pts,
        colors=colors,
        opacity=opacities,
        scales=scales,
        rotations=rotations,
        scale_modifier=scale_modifier,
        viewmatrix=camera_params['view_matrix'],
        projmatrix=camera_params['full_proj_matrix'],
        tan_fovx=camera_params['tan_fovx'],
        tan_fovy=camera_params['tan_fovy'],
        image_height=image_height,
        image_width=image_width,
        sh=shs,
        degree=sh_degree,
        campos=camera_params['camera_center'],
        prefiltered=prefiltered,
        antialiasing=antialiasing,
        clamped=clamped,
        debug=False
    )

    print("Rendering completed")
    
    # Convert the rendered image from device to host
    rendered_array = wp.to_torch(rendered_image).cpu().numpy()
    
    # Display and save using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(rendered_array)
    plt.axis('off')
    plt.savefig("example_render.png", bbox_inches='tight', dpi=150)
    print("Rendered image saved to example_render.png")

