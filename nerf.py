import os 
import time 
import torch 
import torch.nn as nn
import cv2 
import numpy as np
from utils.dataload import load_data, RayGenerator
from torch.utils.tensorboard import SummaryWriter
from utils.nets import Nerf
from utils.xyz import * 
from utils.rendering import *
import argparse 
import yaml 
from tqdm import tqdm 

def img_mse(gt, pred):
	if not torch.is_tensor(gt):
		gt = torch.from_numpy(gt).float()
	return torch.mean((pred - gt)**2)

def img_psnr(gt, pred):
	ten = torch.tensor(10)
	if not torch.is_tensor(gt):
		gt = torch.from_numpy(gt).float()
	psnr = 20*torch.log(torch.max(gt))/torch.log(ten) - 10 * torch.log(img_mse(gt, pred))/torch.log(ten)
	return psnr 

def train(params):
	if not os.path.exists(os.path.join(params['savepath'], params['exp_name'])):
		os.makedirs(os.path.join(params['savepath'], params['exp_name']))
	writer = SummaryWriter('logs/run_{}/'.format(str(time.time())[-10:]))
	batch_size = params['batch_size']
	rg = RayGenerator(params['datapath'], params['half_res'], params['num_train_imgs'])
	train_imgs = torch.stack([torch.from_numpy(s['img']) for s in rg.samples['train']]).reshape(-1,3)

	## exponential Lr decay factor  
	lr_start = params['lr_init']
	lr_end = params['lr_final']
	decay = np.exp(np.log(lr_end / lr_start) / params['num_iters'])
	
	net = Nerf().cuda()
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
	## TODO: Add code to load state dict from pre-trained model
	for i in tqdm(range(params['num_iters'])):
		## main training loop 
		rays, ray_ids = rg.select(mode='train', N=batch_size)
		# rays, ray_ids = rg.select_imgs(mode='train', N=batch_size, im_idxs=[0])
		gt_colors = train_imgs[ray_ids,:].float().cuda()
		optimizer.zero_grad()
		rgb, depth, alpha, acc, w = render_nerf(rays.cuda(), net, params['Nf'])
		loss = criterion(rgb, gt_colors)
		
		loss.backward()
		optimizer.step()
		for p in optimizer.param_groups:
			p['lr'] = p['lr']*decay

		## checkpointing and logging code 
		if i % params['ckpt_loss'] == 0:
			writer.add_scalar("Loss/train", loss.item(), i+1)
			writer.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], i+1)
			print(f'loss: {loss.item()} | epoch: {i+1} ')
		
		if i % params['ckpt_images'] == 0:
			print("--- rendering image ---")
			for ii in params['val_idxs']:
				rgb_img, depth_img, gt_img = render_image(net, rg, batch_size=16000,\
														  im_idx=ii, im_set='train')
				writer.add_images(f'train/RGB_{ii}', rgb_img, global_step=i+1, dataformats='NHWC')
				writer.add_images(f'train/Depth_{ii}', depth_img, global_step=i+1, dataformats='NHWC')
				writer.add_images(f'train/GT_{ii}', gt_img, global_step=i+1, dataformats='NHWC')
				writer.add_scalar(f"Loss/Train_Img_MSE_{ii}", img_mse(gt_img, rgb_img), i+1)
				writer.add_scalar(f"Loss/Train_Img_PSNR_{ii}", img_psnr(gt_img, rgb_img), i+1)

				rgb_img, depth_img, gt_img = render_image(net, rg, batch_size=16000,\
														  im_idx=ii, im_set='val')
				writer.add_images(f'Val/RGB{ii}', rgb_img, global_step=i+1, dataformats='NHWC')
				writer.add_images(f'Val/Depth{ii}', depth_img, global_step=i+1, dataformats='NHWC')
				writer.add_images(f'Val/GT{ii}', gt_img, global_step=i+1, dataformats='NHWC')
				writer.add_scalar(f"Loss/Val_Img_MSE{ii}", img_mse(gt_img, rgb_img), i+1)
				writer.add_scalar(f"Loss/Val_Img_PSNRf{ii}", img_psnr(gt_img, rgb_img), i+1)

		if i% params['ckpt_model'] == 0:
			print("saving model")
			tstamp = str(time.time())
			torch.save(net.state_dict(), os.path.join(params['savepath'], params['exp_name'], tstamp+'.pth'))

	print("saving final model")
	tstamp = str(time.time())
	torch.save(net.state_dict(), os.path.join(params['savepath'], params['exp_name'], tstamp+'.pth'))

	
if __name__=="__main__":
	parser = argparse.ArgumentParser(description='NeRF scene')
	parser.add_argument('--config_path', type=str, default='./configs/lego.yaml',
						help='location of data for training')
	args = parser.parse_args()

	with open(args.config_path) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)
	train(params)

