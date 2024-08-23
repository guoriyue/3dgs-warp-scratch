import cv2
import numpy as np
import torch
import warp as wp
import warnings
from wp_utils.wp_funcs import *

def gamma(x, L=4):	
	assert torch.is_tensor(x), "input needs to be a torch tensor"
	if torch.any(x < -1) or torch.any(x > 1):
		warnings.warn("input not in range -1,1, check rescaling")
	vec = []
	for i in range(L):
		vec += [torch.sin(2**i * x), torch.cos(2**i * x)]
	vec = torch.cat(vec, axis=1)
	return vec

def positional_encoder(vec, Lp=10, Ld=4):
	""" applies positional encoding to input vector vec (5x1) 
	Args:
		vec (torch tensor: Bx6): x, y, z, d1, d2, d3
		Lx: number of levels for x, y, z
		Ld: number of levels for d1, d2, d3 
	Returns:
		posx, posd (torch tensors): B x 3*2*Lx, B x 3*2*Ld
	"""
	x,y,z,d1,d2,d3 = vec[:,0:1], vec[:,1:2], vec[:,2:3], vec[:,3:4], vec[:,4:5], vec[:,5:6] 
	pos_x = gamma(x, Lp)
	pos_y = gamma(y, Lp)
	pos_z = gamma(z, Lp)
	pos_d1 = gamma(d1, Ld)
	pos_d2 = gamma(d2, Ld)
	pos_d3 = gamma(d3, Ld)

	posx = torch.cat([x, y, z, pos_x, pos_y, pos_z], axis=1)
	posd = torch.cat([d1, d2, d3, pos_d1, pos_d2, pos_d3], axis=1)

	return posx, posd

def wp_rays_single_cam(cam_params):
	""" takes in camera params H,W,f returns H*W ray directions with origin 0,0,0
	Args:
		cam_params (list): [H, W, f]
	Returns:
		rays (torch Tensor): 3 x HW
	"""

	# torch.Size([400, 400]) torch.Size([400, 400])
	# torch.Size([3, 400, 400])

	H, W, f  = cam_params
	Hl = torch.arange(H) - H//2
	Wl = torch.arange(W) - W//2
	wp_Hl = wp_arange(H)
	wp_Hl = wp_add_1d_scalar(wp_Hl, -H//2)
	wp_Wl = wp_arange(W)
	wp_Wl = wp_add_1d_scalar(wp_Wl, -W//2)
	grid_x, grid_y = torch.meshgrid(Wl, Hl)

	wp_Wl = wp.array(wp_Wl, dtype=WP_FLOAT32)
	wp_Hl = wp.array(wp_Hl, dtype=WP_FLOAT32)
	wp_grid_x, wp_grid_y = wp_meshgrid(wp_Wl, wp_Hl)

	

	wp_grid_x = wp_to_float32_2d(wp_grid_x)
	wp_grid_y = wp_to_float32_2d(wp_grid_y)

	to_stack = [wp_dot_prod_2d_scalar(wp_grid_x, 1.0/f), wp_dot_prod_2d_scalar(wp_grid_y, -1.0/f), wp_dot_prod_2d_scalar(wp.ones_like(wp_grid_x), -1)]

	wp_rays = wp_stack_2d(to_stack)
	
	rays = torch.stack((grid_x/f, -grid_y/f, -1*torch.ones_like(grid_x))).float()

	assert torch.allclose(rays, wp.to_torch(wp_rays).cpu(), atol=1e-5), "rays not equal"

	rays = rays.permute(0,2,1)
	print(rays)
	rays = torch.reshape(rays, (3,-1)) # 640K ray directions (if H,W = 800)

	wp_rays = wp_permute_3d(wp_rays, [0,2,1])
	print(wp.to_torch(wp_rays))
	wp_rays = wp_rays.reshape(-1, 3)
	return rays


def polar_to_mat(theta):
	""" use r, theta phi to generate transformation matrix for generating pose of camera w.r.t world """
	theta_mat = np.array([[1., 0., 0., 0.],
						  [0., np.cos(theta), np.sin(theta), 0.],
						  [0., -np.sin(theta), np.cos(theta), 0.],
						  [0., 0., 0., 1.]])
	return theta_mat 

def phi_to_mat(phi):
	mat = np.array([[np.cos(phi), np.sin(phi), 0., 0.],
					[-np.sin(phi), np.cos(phi), 0., 0.],
					[0., 0., 1., 0],
					[0., 0., 0., 1. ]])
	return mat 

def spherical_to_pose(r, theta, phi):
	""" returns pose corresponding to the input spherical coordinates r, theta, phi of the camera """
	theta = np.radians(theta)
	phi = np.radians(phi)
	trans_mat = np.array([[1., 0., 0., 0.],
						  [0., 1., 0., 0.],
						  [0., 0., 1., r],
						  [0., 0., 0., 1.]])
	theta_mat = polar_to_mat(theta)
	phi_mat = phi_to_mat(phi)
	pose = phi_mat@theta_mat@trans_mat
	return pose

def poses_to_render(r, theta, n_phi=40):
	""" list of cam poses (torch.Tensor) at radisus r, altitude theta and n_phi continuous azimuths
	Args:
		r: camera distance from origin 
		theta (degrees): altitude of camera (zenith)
	"""
	phis = np.linspace(0, 360.0, n_phi)
	poses = [torch.from_numpy(spherical_to_pose(r, theta, phi)).float() for phi in phis]
	return poses