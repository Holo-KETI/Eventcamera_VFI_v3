import torch
import sys
import os
sys.path.append(os.getcwd())
from thop import profile

from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
	def __init__(self, in_channel, base_channel):
		super().__init__()
		self.conv_in = nn.Sequential(
			nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
		)
		self.conv_block = nn.Sequential(
			nn.LeakyReLU(0.25),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.25),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.25),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.25),
		)

	def forward(self, x):
		x_ = self.conv_in(x)
		x_res = self.conv_block(x_)
		return x_res+x_



class ResBlockIF(nn.Module):
	def __init__(self, in_channel, base_channel):
		super().__init__()
		self.conv_in = nn.Sequential(
			nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
		)
		self.conv_block = nn.Sequential(
			nn.LeakyReLU(0.25),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.25),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.25),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.25),
		)
		self.conv_block1 = nn.Sequential(
			nn.LeakyReLU(0.25),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.25),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.25),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.25),
		)

	def forward(self, x):
		x_ = self.conv_in(x)
		x_res = self.conv_block(x_)+x_
		x_res1 = self.conv_block1(x_res)+x_res
		return x_res1


class BackwardWarp_onnx(nn.Module):
	def __init__(self):
		super(BackwardWarp_onnx, self).__init__()

	def forward(self, img, flow, H, W):
		u = flow[:, 0, :, :]
		v = flow[:, 1, :, :]

		grid_x = torch.arange(W, dtype=torch.float32, device=flow.device).unsqueeze(0).unsqueeze(0).expand(1, H, W)
		grid_y = torch.arange(H, dtype=torch.float32, device=flow.device).unsqueeze(0).unsqueeze(2).expand(1, H, W)

		x = grid_x + u
		y = grid_y + v

		# normalize to [-1, 1]
		x = 2 * (x / (W - 1) - 0.5)
		y = 2 * (y / (H - 1) - 0.5)

		# grid = torch.stack((x, y), dim=3)
		return bilinear_grid_sample_onnx(img, x,y, H, W)


def bilinear_grid_sample_onnx(img, x,y, H, W):
	_, C, _, _ = img.shape


	x = ((x + 1) * 0.5) * (W - 1)
	y = ((y + 1) * 0.5) * (H - 1)

	x0 = torch.floor(x)
	x1 = x0 + 1
	y0 = torch.floor(y)
	y1 = y0 + 1

	x0c = x0.clamp(0, W - 1).long()
	x1c = x1.clamp(0, W - 1).long()
	y0c = y0.clamp(0, H - 1).long()
	y1c = y1.clamp(0, H - 1).long()

	# weights
	wa = (x1 - x) * (y1 - y)
	wb = (x1 - x) * (y - y0)
	wc = (x - x0) * (y1 - y)
	wd = (x - x0) * (y - y0)

	# (B, C, H, W) → (B, H, W, C)
	img_hwcn = img.permute(0, 2, 3, 1)

	img_flat = img_hwcn.reshape(1, -1, C)

	def gather_pixel(ix, iy):
		index = (iy * W + ix).reshape(1, -1, 1)		
		return torch.gather(img_flat, 1, index.expand(1, -1, C))

	Ia = gather_pixel(x0c, y0c)
	Ib = gather_pixel(x0c, y1c)
	Ic = gather_pixel(x1c, y0c)
	Id = gather_pixel(x1c, y1c)

	# weight shape 맞추기
	wa = wa.reshape(1, -1, 1)
	wb = wb.reshape(1, -1, 1)
	wc = wc.reshape(1, -1, 1)
	wd = wd.reshape(1, -1, 1)

	out = Ia * wa + Ib * wb + Ic * wc + Id * wd  # (B, H*W, C)

	out = out.reshape(1, H, W, C)

	out = out.permute(0, 3, 1, 2)

	return out


class FlowDecoder(nn.Module):
	def __init__(self, in_channel, base_channel, conv_base_channel):
		super().__init__()
		self.block_1 = ResBlock(in_channel, base_channel)
		
		self.block_3 = ResBlock(base_channel*2+2, conv_base_channel)
		self.block_4 = ResBlock(base_channel*4+2, conv_base_channel)
		
		self.conv_out2 = nn.Conv2d(conv_base_channel, 2, 3, 1, 1, bias=True)
		self.conv_out3 = nn.Conv2d(conv_base_channel, 2, 3, 1, 1, bias=True)
		self.backwarp = BackwardWarp_onnx()

	def forward(self, events, im1, last_flow, last_warp_res, prev_econv):
		
		encoder_in = torch.cat((last_warp_res, events, prev_econv), 1) #1,384,144,232
	
		im_tiplus = self.block_1(encoder_in) #1,128,144,232

		border_flow_in = torch.cat((last_flow, im1, im_tiplus), 1) #1, 258, 144, 232
		block_3 = self.block_3(border_flow_in) #1,128,144,232

		flow0 = self.conv_out2(block_3)+last_flow #1, 2, 144, 232
		warp_res = self.backwarp(im1, flow0, 144, 232) #1, 128, 144, 232

		final_flow_in = torch.cat((im_tiplus, im1, flow0, warp_res, block_3), 1) #1, 514, 144, 232
		block_4 = self.block_4(final_flow_in) #1, 128, 144, 232
		flow = self.conv_out3(block_4)+flow0 #1, 2, 144, 232
	
		out_warp_res = self.backwarp(im1, flow,144, 232) #1, 128, 144, 232
		
		flow_large = F.interpolate(flow, scale_factor=4, mode='bilinear')*4 #1, 2, 576, 928
		return out_warp_res, flow, flow_large, im_tiplus