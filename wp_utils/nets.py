import numpy as np
# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
import warp as wp
import warnings
from utils.xyz import *

class Nerf(nn.Module):
	def __init__(self, Lp = 10, Ld = 4, H=256):
		super(Nerf, self).__init__()
		self.Ld = Ld 
		self.Lp = Lp
		in_Cx = Lp*6 +3
		in_Cd = Ld*6 +3
		layers = []
		layers = [nn.Linear(in_Cx, H), nn.ReLU()]
		for i in range(4):
			layers += [nn.Linear(H, H), nn.ReLU()]
		self.layers_0 = nn.Sequential(*layers)

		self.skip_conn_layer = nn.Sequential(nn.Linear(H+in_Cx, H), nn.ReLU())

		self.layers_1 = nn.Sequential(nn.Linear(H,H),
									  nn.ReLU(),
									  nn.Linear(H,H),
									  nn.ReLU())
		self.sigma_fc = nn.Sequential(nn.Linear(H,1))
		self.layers_2 = nn.Linear(H,H)

		self.color_fc = nn.Sequential(nn.Linear(H+in_Cd, H//2),
									  nn.ReLU(),
									  nn.Linear(H//2,3))

	def forward(self, v):
		# v: B x 6
		x, d = positional_encoder(v, Lp=self.Lp, Ld=self.Ld)
		out = self.layers_0(x)
		out = self.skip_conn_layer(torch.cat([out, x],axis=1))
		out = self.layers_1(out)
		sigma = self.sigma_fc(out)
		out = self.layers_2(out)
		color = self.color_fc(torch.cat((out, d),axis=1))
		return torch.cat((color, sigma),axis=1) 

class CoarseNet(nn.Module):
	pass

class FineNet(nn.Module):
	pass