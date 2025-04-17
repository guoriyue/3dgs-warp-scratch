#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@ Description:
@ Date     : 2024/05/20 17:20:00
@ Author   : sunyifan
@ Version  : 1.0
"""

import math
import numpy as np
from tqdm import tqdm
from loguru import logger
from math import sqrt, ceil

from render_python import computeColorFromSH
from render_python import computeCov2D, computeCov3D
from render_python import transformPoint4x4, in_frustum
from render_python import getWorld2View2, getProjectionMatrix, ndc2Pix, in_frustum


class Rasterizer:
    def __init__(self) -> None:
        pass

    def forward(
        self,
        P,  # int, num of guassians
        D,  # int, degree of spherical harmonics
        M,  # int, num of sh base function
        background,  # color of background, default black
        width,  # int, width of output image
        height,  # int, height of output image
        means3D,  # ()center position of 3d gaussian
        shs,  # spherical harmonics coefficient
        colors_precomp,
        opacities,  # opacities
        scales,  # scale of 3d gaussians
        scale_modifier,  # default 1
        rotations,  # rotation of 3d gaussians
        cov3d_precomp,
        viewmatrix,  # matrix for view transformation
        projmatrix,  # *(4, 4), matrix for transformation, aka mvp
        cam_pos,  # position of camera
        tan_fovx,  # float, tan value of fovx
        tan_fovy,  # float, tan value of fovy
        prefiltered,
    ) -> None:

        focal_y = height / (2 * tan_fovy)  # focal of y axis
        focal_x = width / (2 * tan_fovx)

        # run preprocessing per-Gaussians
        # transformation, bounding, conversion of SHs to RGB
        logger.info("Starting preprocess per 3d gaussian...")
        preprocessed = self.preprocess(
            P,
            D,
            M,
            means3D,
            scales,
            scale_modifier,
            rotations,
            opacities,
            shs,
            viewmatrix,
            projmatrix,
            cam_pos,
            width,
            height,
            focal_x,
            focal_y,
            tan_fovx,
            tan_fovy,
        )
        

        # produce [depth] key and corresponding guassian indices
        # sort indices by depth
        depths = preprocessed["depths"]
        point_list = np.argsort(depths)
        
        # Print preprocessed data in a more readable format
        print("\n" + "="*80)
        print(" "*30 + "PREPROCESSED DATA SUMMARY")
        print("="*80)
        
        print(f"\nImage dimensions: {width} x {height}")
        print(f"Number of Gaussians: {len(point_list)}")
        print(f"Background color: {background}")
        
        print("\n" + "-"*40)
        print("Point list (sorted by depth):")
        print("-"*40)
        print(point_list)
        
        print("\n" + "-"*40)
        print("2D Points in image space (first 3):")
        print("-"*40)
        if len(preprocessed["points_xy_image"]) > 0:
            for i in range(min(3, len(preprocessed["points_xy_image"]))):
                print(f"  Point {i}: {preprocessed['points_xy_image'][i]}")
        else:
            print("  No points available")
        
        print("\n" + "-"*40)
        print("RGB values (first 3):")
        print("-"*40)
        if len(preprocessed["rgbs"]) > 0:
            for i in range(min(3, len(preprocessed["rgbs"]))):
                print(f"  RGB {i}: {preprocessed['rgbs'][i]}")
        else:
            print("  No RGB values available")
        
        print("\n" + "-"*40)
        print("Conic and opacity values (first 3):")
        print("-"*40)
        if len(preprocessed["conic_opacity"]) > 0:
            for i in range(min(3, len(preprocessed["conic_opacity"]))):
                conic = preprocessed["conic_opacity"][i]
                print(f"  Conic {i}: [{float(conic[0]):.4f}, {float(conic[1]):.4f}, {float(conic[2]):.4f}], Opacity: {float(conic[3]):.4f}")
        else:
            print("  No conic values available")
        
        print("\n" + "-"*40)
        print("Other preprocessed data:")
        print("-"*40)
        for key in preprocessed:
            if key not in ["points_xy_image", "rgbs", "conic_opacity", "depths"]:
                data = preprocessed[key]
                if isinstance(data, list) or isinstance(data, np.ndarray):
                    print(f"  {key}: shape={np.array(data).shape}")
                else:
                    print(f"  {key}: {data}")
        print("depths", preprocessed["depths"])
        print("\n" + "="*80)
        print(" "*35 + "END SUMMARY")
        print("="*80 + "\n")
        # exit()

        # render
        logger.info("Starting render...")
        out_color = self.render(
            point_list,
            width,
            height,
            preprocessed["points_xy_image"],
            preprocessed["rgbs"],
            preprocessed["conic_opacity"],
            background,
        )
        return out_color

    def preprocess(
        self,
        P,
        D,
        M,
        orig_points,
        scales,
        scale_modifier,
        rotations,
        opacities,
        shs,
        viewmatrix,
        projmatrix,
        cam_pos,
        W,
        H,
        focal_x,
        focal_y,
        tan_fovx,
        tan_fovy,
    ):

        rgbs = []  # rgb colors of gaussians
        cov3Ds = []  # covariance of 3d gaussians
        depths = []  # depth of 3d gaussians after view&proj transformation
        radii = []  # radius of 2d gaussians
        conic_opacity = []  # covariance inverse of 2d gaussian and opacity
        points_xy_image = []  # mean of 2d guassians
        for idx in range(P):
            # make sure point in frustum
            p_orig = orig_points[idx]
            print(f"p_orig: {p_orig}")
            p_view = in_frustum(p_orig, viewmatrix)
            print(f"p_view: {p_view}")
            if p_view is None:
                continue
            depths.append(p_view[2])
            print(f"depths: {depths}")

            # transform point, from world to ndc
            # Notice, projmatrix already processed as mvp matrix
            p_hom = transformPoint4x4(p_orig, projmatrix)
            print(f"p_hom: {p_hom}")
            p_w = 1 / (p_hom[3] + 0.0000001)
            print(f"p_w: {p_w}")
            p_proj = [p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w]
            print(f"p_proj: {p_proj}")

            # compute 3d covarance by scaling and rotation parameters
            scale = scales[idx]
            rotation = rotations[idx]
            cov3D = computeCov3D(scale, scale_modifier, rotation)
            cov3Ds.append(cov3D)
            
            print(f"cov3D: {cov3D}")

            # compute 2D screen-space covariance matrix
            # based on splatting, -> JW Sigma W^T J^T
            cov = computeCov2D(
                p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix
            )
            print(f"cov: {cov}")
            # exit()

            # invert covarance(EWA splatting)
            det = cov[0] * cov[2] - cov[1] * cov[1]
            if det == 0:
                depths.pop()
                cov3Ds.pop()
                continue
            det_inv = 1 / det
            conic = [cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv]
            conic_opacity.append([conic[0], conic[1], conic[2], opacities[idx]])
            print("cov: ", cov)
            # compute radius, by finding eigenvalues of 2d covariance
            # transfrom point from NDC to Pixel
            mid = 0.5 * (cov[0] + cov[1])
            lambda1 = mid + sqrt(max(0.1, mid * mid - det))
            lambda2 = mid - sqrt(max(0.1, mid * mid - det))
            print("lambda1: ", lambda1)
            print("lambda2: ", lambda2)
            my_radius = ceil(3 * sqrt(max(lambda1, lambda2)))
            point_image = [ndc2Pix(p_proj[0], W), ndc2Pix(p_proj[1], H)]
            print(f"point_image: {point_image}")
            print(f"my_radius: {my_radius}")
            radii.append(my_radius)
            points_xy_image.append(point_image)

            # convert spherical harmonics coefficients to RGB color
            sh = shs[idx]
            print(f"sh: {sh}")
            result = computeColorFromSH(D, p_orig, cam_pos, sh)
            print(f"result: {result}")
            # exit()
            rgbs.append(result)

        return dict(
            rgbs=rgbs,
            cov3Ds=cov3Ds,
            depths=depths,
            radii=radii,
            conic_opacity=conic_opacity,
            points_xy_image=points_xy_image,
        )

    def render(
        self, point_list, W, H, points_xy_image, features, conic_opacity, bg_color
    ):

        out_color = np.zeros((H, W, 3))
        pbar = tqdm(range(H * W))

        # loop pixel
        for i in range(H):
            for j in range(W):
                pbar.update(1)
                pixf = [i, j]
                C = [0, 0, 0]

                # loop gaussian
                for idx in point_list:

                    # init helper variables, transmirrance
                    T = 1

                    # Resample using conic matrix
                    # (cf. "Surface Splatting" by Zwicker et al., 2001)
                    xy = points_xy_image[idx]  # center of 2d gaussian
                    d = [
                        xy[0] - pixf[0],
                        xy[1] - pixf[1],
                    ]  # distance from center of pixel
                    con_o = conic_opacity[idx]
                    power = (
                        -0.5 * (con_o[0] * d[0] * d[0] + con_o[2] * d[1] * d[1])
                        - con_o[1] * d[0] * d[1]
                    )
                    if power > 0:
                        continue

                    # Eq. (2) from 3D Gaussian splatting paper.
                    # Compute color
                    alpha = min(0.99, con_o[3] * np.exp(power))
                    if alpha < 1 / 255:
                        continue
                    test_T = T * (1 - alpha)
                    if test_T < 0.0001:
                        break

                    # Eq. (3) from 3D Gaussian splatting paper.
                    color = features[idx]
                    for ch in range(3):
                        C[ch] += color[ch] * alpha * T

                    T = test_T

                # get final color
                for ch in range(3):
                    out_color[j, i, ch] = C[ch] + T * bg_color[ch]
                
                # print("j, i, out_color[j, i, :]", j, i, out_color[j, i, :])
                if out_color[j, i, 0] != 0.0:
                    print("out_color[j, i, :] j, i", j, i, out_color[j, i, :])

        return out_color


if __name__ == "__main__":
    # set guassian
    pts = np.array([[2, 0, -2], [0, 2, -2], [-2, 0, -2]])
    n = len(pts)
    # Create fixed interval SH coefficients from 0 to 1
    # shs = np.random.random((n, 16, 3))
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
    opacities = np.ones((n, 1))
    scales = np.ones((n, 3))
    rotations = np.array([np.eye(3)] * n)

    # set camera
    cam_pos = np.array([0, 0, 5])
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
    viewmatrix = getWorld2View2(R=R, t=cam_pos)
    projmatrix = getProjectionMatrix(**proj_param)
    projmatrix = np.dot(projmatrix, viewmatrix)
    tanfovx = math.tan(proj_param["fovX"] * 0.5)
    tanfovy = math.tan(proj_param["fovY"] * 0.5)

    # render
    rasterizer = Rasterizer()
    
    # Print all parameters for debugging and comparison with render.py
    print("\n----- 3DGS DEBUGGING PARAMETERS -----")
    print(f"Image dimensions: {700}x{700}")
    print(f"Background color: {np.array([0, 0, 0])}")
    print(f"Number of Gaussians: {len(pts)}")
    print(f"Gaussian means shape: {pts.shape}")
    print(f"SH coefficients shape: {shs.shape}")
    print(f"Gaussian scales shape: {scales.shape}")
    print(f"Gaussian rotations shape: {rotations.shape}")
    print(f"Gaussian opacities shape: {opacities.shape}")
    print(f"Scale modifier: {1}")
    print(f"Tan fovx: {tanfovx}, Tan fovy: {tanfovy}")
    print(f"Focal length x: {700/(2*tanfovx)}, Focal length y: {700/(2*tanfovy)}")
    print(f"Camera position: {cam_pos}")
    
    # Print some sample values for comparison
    if len(pts) > 0:
        print("\nSample values for first Gaussian:")
        print(f"  Position: {pts[0]}")
        print(f"  SH coeffs (first few): {shs[0][0:3]}")
        print(f"  Scale: {scales[0]}")
        print(f"  Rotation: {rotations[0]}")
        print(f"  Opacity: {opacities[0]}")
    
    # Print view and projection matrices
    print("\nView matrix:")
    print(viewmatrix)
    print("\nProjection matrix:")
    print(projmatrix)
    print("----- END 3DGS DEBUGGING PARAMETERS -----\n")
    
    
    
    out_color = rasterizer.forward(
        P=len(pts),
        D=3,
        M=16,
        background=np.array([0, 0, 0]),
        width=700,
        height=700,
        means3D=pts,
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        scale_modifier=1,
        rotations=rotations,
        cov3d_precomp=None,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        cam_pos=cam_pos,
        tan_fovx=tanfovx,
        tan_fovy=tanfovy,
        prefiltered=None,
    )

    import matplotlib.pyplot as plt
    plt.imshow(out_color)
    plt.show()
