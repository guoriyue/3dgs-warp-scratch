import numpy as np
import cv2 
import os 
import time
from tqdm import tqdm
# import torch.nn.functional as F
# import torch.nn as nn
import warp as wp
from utils.xyz import positional_encoder
from utils.xyz import rays_single_cam 

def render_nerf(rays, net, N, tn=2, tf=6):
	""" stratified sampling on a set of rays using Nerf model
	Args:
		rays (torch Tensor): B x 6 
		net (nerf model):
	Returns:
		 out: BxNx4	
	"""
	## input to nerf model BN x 6 
	## Nerf output - BN x 4  --> reshape BxNx4 
	## t sample dims - BxN
	B = rays.size(0)
	t_bins = torch.linspace(tn,tf,N+1)
	bin_diff = t_bins[1] - t_bins[0] 

	unif_samps = torch.rand(rays.size(0),N)
	ts = bin_diff* unif_samps + t_bins[:-1] # BxN 
	ts = ts.cuda()
	origins = rays[:,:3] # Bx3
	dirs = rays[:,3:]  # Bx3

	disp = dirs.unsqueeze(-1)*ts.unsqueeze(1) # Bx1x3 * BxNx1 = Bx3xN
	
	locs = origins.unsqueeze(-1) + disp # Bx3x1 + Bx3xN = Bx3xN 
	dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)	
	query_pts = torch.cat((locs, dirs.unsqueeze(-1).expand(-1,-1,N)),dim=1) # Bx6xN
	query_pts = query_pts.permute(0,2,1) # BxNx6
	query_pts = query_pts.reshape(-1,6)
	out = net.forward(query_pts)
	out = out.reshape(B,N,4)
	rgb_v, depth_v, alpha_v, acc_v, w_v = volume_render(out, ts, dirs)

	return rgb_v, depth_v, alpha_v, acc_v, w_v

def volume_render(nerf_outs, ts, dirs):
	""" computes color, depth and alphas along rays using NeRF outputs (section 4)
	Args:
		ts (torch tensor) BxN | B:number of rays, N: number of samples along a ray 
		nerf_outs (torch tensor)BxNx4 | RGB \sigma (4 values) for each sample along each ray 
	Returns: 
		rgb (torch tensor) Bx3
		depth (torch tensor) (B,)
		alphas (torch tensor) (B,N)
		acc (torch tensor) (B,)
		weights (torch tensor) (B,N) 
	"""

	deltas = ts[:,1:] - ts[:,:-1]
	deltas = torch.cat((deltas, 1e10*torch.ones_like(deltas[:,:1])), dim=1)
	deltas = deltas * torch.norm(dirs[...,None,:], dim=-1)
	## TODO add gaussian noise regularizer 

	sigma = nerf_outs[...,3]

	alpha = 1 - torch.exp(-nn.functional.softplus(sigma)*deltas) 
	weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1]
	
	# ## Eqn 5 of paper w: BxN
	# weights = alpha*T 
	
	## accumulated rgb color along each ray (Bx3)
	rgb = torch.sum(weights.unsqueeze(-1)*nerf_outs[...,:3], axis=1)
	## depth along each ray (B,), weighted average of samples 
	depth = torch.sum(weights * ts, axis=-1)

	## accumulation map (B,): average of weight values along a ray 
	acc = torch.sum(weights, axis=-1)
	
	## computing disparity from depth 
	disp = torch.max(1e-10 * torch.ones_like(depth), depth / torch.sum(weights, dim=-1))
	disp = 1. / disp

	return rgb, disp, alpha, acc, weights


def render_image(net, rg, batch_size=64000, im_idx=0, im_set='val'):
	""" render an image and depth map from val set (currently hardcoded) from trained NeRF model """

	gt_img = rg.samples[im_set][im_idx]['img']
	H,W = gt_img.shape[0], gt_img.shape[1]

	NUM_RAYS = H*W # number of rays in image, currently hardcoded
	net = net.cuda()
	rays = rg.rays_dataset[im_set][im_idx*NUM_RAYS:(im_idx+1)*NUM_RAYS,:]
	rgbs = [] 
	depths = [] 
	with torch.no_grad():
		for i in tqdm(range(rays.size(0) // batch_size)):
			inp_rays = rays[i*batch_size:(i+1)*batch_size]
			rgb, depth, _, _, _ = render_nerf(inp_rays.cuda(), net, N=128)
			rgb = torch.clip(rgb, torch.tensor(0.).cuda(), torch.tensor(1.).cuda())
			rgbs.append(rgb)
			depths.append(depth)

	rgb = torch.cat(rgbs).cpu()
	depth = torch.cat(depths).cpu()
	
	rgb_img = rgb.reshape(1,H,W,3) ## permuting for tensorboard
	depth_img = depth.reshape(1,H,W,1) ## permuting for tensorboard
	gt_img = gt_img.reshape(1,H,W,3)
	return rgb_img, depth_img, gt_img


def render_poses(net, poses, cam_params, batch_size, savepath=''):
	""" render an image and depth map from a given set of poses 
	Args:
		net: instance of Nerf() model loaded with params 
		poses: list of 4x4 (np.array) camera poses in which scene is to be rendered
		cam_params: [H, W, f]
	Returns:
		An animation of poses 
	"""
	H,W = cam_params[0], cam_params[1]
	num_imgs = len(poses)
	NUM_RAYS = H*W # number of rays in image
	net = net.cuda()
	rays_1_cam = rays_single_cam(cam_params)
	transf_mats = torch.stack(poses)
	rays_dataset = torch.matmul(transf_mats[:,:3,:3], rays_1_cam)
	cam_origins = transf_mats[:,:3,3:]
	cam_origins = cam_origins.expand(len(poses),3,H*W) #Bx3xHW
	rays_all = torch.cat((cam_origins, rays_dataset),dim=1).permute(0,2,1).reshape(-1, 6) # BHW x 6, number of cameras 

	rgb_imgs = []
	depth_imgs = []
	with torch.no_grad():
		for idx in range(num_imgs):
			rgbs = [] 
			depths = [] 
			rays = rays_all[NUM_RAYS*idx:NUM_RAYS*(idx+1), :]
			for i in tqdm(range(rays.size(0) // batch_size)):
				inp_rays = rays[i*batch_size:(i+1)*batch_size]
				rgb, depth, _, _, _ = render_nerf(inp_rays.cuda(), net, N=128)
				rgb = torch.clip(rgb, torch.tensor(0.).cuda(), torch.tensor(1.).cuda())
				rgbs.append(rgb)
				depths.append(depth)

			rgb = torch.cat(rgbs).cpu().reshape(H,W,3).numpy()
			depth = torch.cat(depths).cpu().reshape(H,W).numpy()
			rgb_imgs.append(rgb)
			depth_imgs.append(depth)
	
	tstamp = str(time.time())
	out = cv2.VideoWriter(os.path.join(savepath,f'nerf_rgb{tstamp[-10:]}.mp4'),cv2.VideoWriter_fourcc('m','p','4','v'),15, (H,W))
	for i in range(len(rgb_imgs)):
		rgb_img = cv2.cvtColor(rgb_imgs[i], cv2.COLOR_RGB2BGR)
		out.write((rgb_img*255).astype(np.uint8))
	out.release()