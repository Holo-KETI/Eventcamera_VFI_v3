import torch
import sys
import os
sys.path.append(os.getcwd())
import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
	def __init__(self, in_channel, base_channel):
		super().__init__()
		self.conv_in = nn.Sequential(
			nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.25)
		)
		self.conv_block = nn.Sequential(
			nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.25),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.25),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True)
		)

	def forward(self, x):
		x_ = self.conv_in(x)
		x_res = self.conv_block(x)
		return x_res + x_



class ImageEncoder(nn.Module):
	def __init__(self, img_ch, base_channel):
		super().__init__()
		self.img_ch = img_ch
		self.preblock = ResBlock(img_ch, base_channel)
		self.down0 = nn.Conv2d(base_channel, 2*base_channel, 3, 2, 1, bias=True)
		self.down1 = nn.Conv2d(2*base_channel, 4*base_channel, 3, 2, 1, bias=True)
		self.n =1

	def forward(self, img1, img2):
		
		img = torch.cat((img1,img2), dim=1)
		img = torch.cat(img.split(self.img_ch, 1), 0)
		preconv = self.preblock(img)
		im_half = self.down0(preconv)
		im_quater = self.down1(im_half)

		return im_quater[:self.n], im_quater[self.n:], preconv[:self.n], preconv[self.n:]