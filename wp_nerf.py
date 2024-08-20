import os 
import time
import cv2 
import numpy as np
from utils.dataload import load_data, RayGenerator
from torch.utils.tensorboard import SummaryWriter
import torch
# only use torch writer and save
# from utils.nets import WpNerf
# from utils.xyz import * 
from utils.rendering import *
import argparse 
import yaml 
from tqdm import tqdm 
import warp as wp
from warp.optim import Adam

WP_INT = wp.int32
WP_FLOAT32 = wp.float32

@wp.kernel
def wp_stack(head: WP_INT, stack_tensor: wp.array2d(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j + head * stack_tensor.shape[1]] = stack_tensor[i, j]

def train(params):
    if not os.path.exists(os.path.join(params['savepath'], params['exp_name'])):
        os.makedirs(os.path.join(params['savepath'], params['exp_name']))
    writer = SummaryWriter('logs/run_{}/'.format(str(time.time())[-10:]))
    batch_size = params['batch_size']
    rg = RayGenerator(params['datapath'], params['half_res'], params['num_train_imgs'])
    train_imgs = torch.stack([torch.from_numpy(s['img']) for s in rg.samples['train']]).reshape(-1,3)
    # train_imgs_list = []
    # for s in rg.samples['train']:
    #     train_imgs_list.append(wp.from_numpy(s['img']))
    # print(len(train_imgs_list), train_imgs_list[0].shape)
    # train_imgs = wp.zeros(shape=(len(train_imgs_list), train_imgs_list[0].shape[0], train_imgs_list[0].shape[1], train_imgs_list[0].shape[2]), dtype=WP_FLOAT32)
    # # train_imgs = wp_stack(train_imgs).reshape(-1, 3)
    # # train_imgs = torch.stack([torch.from_numpy(s['img']) for s in rg.samples['train']]).reshape(-1,3)
    # wp.launch(wp_stack, [train_imgs.shape[0], train_imgs.shape[1]], [train_imgs.shape[0], train_imgs.shape[1]*params['num_train_imgs']], train_imgs, train_imgs)
    # ## exponential Lr decay factor  
    # lr_start = params['lr_init']
    # lr_end = params['lr_final']
    # decay = np.exp(np.log(lr_end / lr_start) / params['num_iters'])
    # net = WpNerf()
    # criterion = nn.MSELoss()
    # optimizer = Adam(net.parameters(), lr=5e-4)
    # ## TODO: Add code to load state dict from pre-trained model
    # for i in tqdm(range(params['num_iters'])):
    # 	## main training loop 
    # 	rays, ray_ids = rg.select(mode='train', N=batch_size)
    # 	# rays, ray_ids = rg.select_imgs(mode='train', N=batch_size, im_idxs=[0])
    # 	gt_colors = train_imgs[ray_ids,:].float().cuda()
    # 	# optimizer.zero_grad()
    # 	rgb, depth, alpha, acc, w = render_nerf(rays.cuda(), net, params['Nf'])
    # 	loss = criterion(rgb, gt_colors)
    # 	loss.backward()
    # 	optimizer.step(net.parameters_grad())
    # 	for p in optimizer.param_groups:
    # 		p['lr'] = p['lr']*decay
    # 	## checkpointing and logging code 
    # 	if i % params['ckpt_loss'] == 0:
    # 		writer.add_scalar("Loss/train", loss.item(), i+1)
    # 		writer.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], i+1)
    # 		print(f'loss: {loss.item()} | epoch: {i+1} ')
    # 	if i % params['ckpt_images'] == 0:
    # 		print("--- rendering image ---")
    # 		for ii in params['val_idxs']:
    # 			rgb_img, depth_img, gt_img = render_image(net, rg, batch_size=16000,\
    # 													  im_idx=ii, im_set='train')
    # 			writer.add_images(f'train/RGB_{ii}', rgb_img, global_step=i+1, dataformats='NHWC')
    # 			writer.add_images(f'train/Depth_{ii}', depth_img, global_step=i+1, dataformats='NHWC')
    # 			writer.add_images(f'train/GT_{ii}', gt_img, global_step=i+1, dataformats='NHWC')
    # 			rgb_img, depth_img, gt_img = render_image(net, rg, batch_size=16000,\
    # 													  im_idx=ii, im_set='val')
    # 			writer.add_images(f'Val/RGB{ii}', rgb_img, global_step=i+1, dataformats='NHWC')
    # 			writer.add_images(f'Val/Depth{ii}', depth_img, global_step=i+1, dataformats='NHWC')
    # 			writer.add_images(f'Val/GT{ii}', gt_img, global_step=i+1, dataformats='NHWC')
    # 	if i % params['ckpt_model'] == 0:
    # 		print("saving model")
    # 		tstamp = str(time.time())
    # 		torch.save(net.state_dict(), os.path.join(params['savepath'], params['exp_name'], tstamp + _f'{i}.pth'))
    # print("saving final model")
    # tstamp = str(time.time())
    # torch.save(net.state_dict(), os.path.join(params['savepath'], params['exp_name'], tstamp+'.pth'))


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='NeRF scene')
	parser.add_argument('--config_path', type=str, default='./configs/lego.yaml',
						help='location of data for training')
	args = parser.parse_args()

	with open(args.config_path) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)
	train(params)

