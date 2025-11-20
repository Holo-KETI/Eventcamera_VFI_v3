import torch
import sys
import os
sys.path.append(os.getcwd())
from thop import profile

from torch import nn
from torch.nn import functional as F



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
	
class IFBlock(nn.Module):
	def __init__(self, in_channel, base_channel):
		super().__init__()
		self.bwarp = BackwardWarp_onnx()
		self.resConv = ResBlockIF(in_channel, base_channel)
		self.out_conv = nn.Conv2d(base_channel, 5, 3, 1, 1, bias=True)
        
	def forward(self, i0, i1, i0tf, i1tf, ft0, ft1, m, t, scale, forward_feat=None, backward_feat=None):
		
		data_in = torch.cat((i0, i1, i0tf, i1tf), 1)

		data_in = torch.cat((data_in, t), dim=1)
		
		
		
		data_in = torch.cat((data_in, ft0, ft1), 1)
		data_in_lowres = F.interpolate(data_in, scale_factor=scale, mode='bilinear')
		

		data_in_lowres = torch.cat((data_in_lowres, forward_feat, backward_feat), 1)

		## multi-stream ################################################################################

		after_conv_lowres = self.out_conv(self.resConv(data_in_lowres))
		after_conv_highres = F.interpolate(after_conv_lowres, scale_factor=4, mode='bilinear')
		
		Ft0 = ft0+after_conv_highres[:, 1:3]
		Ft1 = ft1+after_conv_highres[:, 3:]

		Mask = after_conv_highres[:, :1]

		################################################################################################

		I0tf = self.bwarp(i0, Ft0, 576, 928)
		I1tf = self.bwarp(i1, Ft1, 576, 928)
		
		return Ft0, Ft1, I0tf, I1tf, Mask

class IFBlock_half(nn.Module):
	def __init__(self, in_channel, base_channel):
		super().__init__()
		self.bwarp = BackwardWarp_onnx()
		self.resConv = ResBlockIF(in_channel, base_channel)
		self.out_conv = nn.Conv2d(base_channel, 5, 3, 1, 1, bias=True)
        
	def forward(self, i0, i1, i0tf, i1tf, ft0, ft1, m, t, scale, forward_feat=None, backward_feat=None):
		data_in = torch.cat((i0, i1, i0tf, i1tf), 1)
	
		data_in = torch.cat((data_in, m), dim=1)
		

		
		data_in = torch.cat((data_in, ft0, ft1), 1)
		data_in_lowres = F.interpolate(data_in, scale_factor=scale, mode='bilinear')
		
		

		after_conv_lowres = self.out_conv(self.resConv(data_in_lowres))
		after_conv_highres = F.interpolate(after_conv_lowres, scale_factor=2, mode='bilinear')
		Ft0 = ft0+after_conv_highres[:, 1:3]
		Ft1 = ft1+after_conv_highres[:, 3:]

		Mask = after_conv_highres[:, :1]+m

	
		#[1, 3, 576, 928] 
		I0tf = self.bwarp(i0, Ft0, 576, 928)

		I1tf = self.bwarp(i1, Ft1, 576, 928)
		
		return Ft0, Ft1, I0tf, I1tf, Mask
	

class MaskNet(nn.Module):
	def __init__(self, in_channel, base_channel):
		super().__init__()
		
		self.sig = nn.Sigmoid()

		self.quater_block = IFBlock(in_channel+2*base_channel, base_channel)
		self.half_block = IFBlock_half(in_channel, base_channel//2)
		self.channel_squeeze0 = nn.Conv2d(128, 32, 1, 1, 0, bias=True)
		self.channel_squeeze1 = nn.Conv2d(128, 32, 1, 1, 0, bias=True)
		self.refine_blocks = ResBlockIF(3+128, 64)
		self.conv_out = nn.Conv2d(64, 3, 1, 1, bias=True)
		self.zero_tensor = torch.zeros(1, 1, 576, 928).cuda()

	def forward(self, im0, im1, i0tf, i1tf, t, f_t0, f_t1, forward_feat, backward_feat, im0_feat_full, im1_feat_full):

		qft0, qft1, qi0tf, qi1tf, qm = self.quater_block(im0, im1, i0tf, i1tf, f_t0, f_t1, None, t, 0.25, forward_feat, backward_feat)
		hft0, hft1, hi0tf, hi1tf, hm = self.half_block(im0, im1, qi0tf, qi1tf, qft0, qft1, qm, t, 0.5)
		mask = self.sig(hm)
		fuse_out = hi0tf*mask+(1-mask)*hi1tf

		forward_feat_ld = self.channel_squeeze0(forward_feat)
		backward_feat_ld = self.channel_squeeze1(backward_feat)
		forward_feat_hd, backward_feat_hd = F.interpolate(forward_feat_ld, scale_factor=4, mode='bilinear'), F.interpolate(backward_feat_ld, scale_factor=4, mode='bilinear')
		
		refine_in = torch.cat((forward_feat_hd, backward_feat_hd, fuse_out, im0_feat_full, im1_feat_full), 1)
		refine_out = self.conv_out(self.refine_blocks(refine_in))+fuse_out
		return refine_out, hft0, hft1