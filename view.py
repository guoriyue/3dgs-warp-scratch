import os 
import time 
import torch 
import torch.nn as nn
import cv2 
import numpy as np
from utils.dataload import load_data, RayGenerator
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from utils.nets import Nerf
from utils.xyz import * 
from utils.rendering import *
# from utils.visualization import render_poses, poses_to_render
import argparse 
import yaml 
from tqdm import tqdm 

def test(params):
	assert os.path.exists(params['loadpath']), "model path doesn't exist"
	if not os.path.exists(os.path.join(params['savepath'], params['exp_name'])):
		os.makedirs(os.path.join(params['savepath'], params['exp_name']))
	
	savepath = os.path.join(params['savepath'], params['exp_name'])
	batch_size = params['batch_size']
	rg = RayGenerator(params['datapath'], params['half_res'])
	cam_params = [rg.H, rg.W, rg.f]
	net = Nerf().cuda()
	net.load_state_dict(torch.load(params['loadpath']),strict=True)
	
	if params['animation']:
		theta = -params['theta']
		n_phi = params['num_poses']
		poses = poses_to_render(r=4, theta=theta, n_phi=n_phi)
		render_poses(net, poses, cam_params, batch_size, savepath)
		return

	im_set = params['im_set'] # can be 'train', 'test', 'val' depending on whcich images to render
	print(f"saving images to {params['savepath']}")
	with torch.no_grad():
		for idx in params['im_idxs']:
			rgb_img, depth_img, gt_img = render_image(net, rg, batch_size=params['batch_size'],\
											 		  im_idx=idx, im_set=im_set)
			rgb_out = torch.cat((torch.from_numpy(gt_img), rgb_img),axis=0)
			save_image(make_grid(rgb_out.permute(0,3,1,2)), os.path.join(params['savepath'], params['exp_name'], f'rgb_{idx}.png'))
			save_image(make_grid(depth_img.permute(0,3,1,2)), os.path.join(params['savepath'], params['exp_name'], f'depth_{idx}.png'))
	
if __name__=="__main__":
	parser = argparse.ArgumentParser(description='NeRF scene')
	parser.add_argument('--config_path', type=str, default='./configs/lego_view.yaml',
						help='location of data for training')
	args = parser.parse_args()

	with open(args.config_path) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)
	test(params)

