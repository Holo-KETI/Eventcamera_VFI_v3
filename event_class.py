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


class EventsEncoder(nn.Module):
	def __init__(self, e_channel, img_chn, base_channel, num_decoder=None):
		super().__init__()
		self.step = num_decoder
		self.cur_step = 12
		self.pre_conv = ResBlock(e_channel+img_chn, e_channel)
		self.ct=  nn.Parameter(torch.ones((1, e_channel, 1, 1))*0.2, requires_grad=True)
		self.down0 = nn.Sequential(
			nn.Conv2d(num_decoder, 2*base_channel, 3, 2, 1, bias=True),
			nn.LeakyReLU(0.25)
		)
		self.down1 = nn.Sequential(
			nn.Conv2d(2*base_channel, 4*base_channel, 3, 2, 1, bias=True),
			nn.LeakyReLU(0.25)
		)
		self.e_channel = e_channel
		self.padding_dict = {}
		self.n = 1
		self.h = 576
		self.w = 928

	def get_padding(self, b, c, h, w, device):
		# k = f"{b}_{c}_{h}_{w}"

		padding = torch.zeros((b, c, h, w), requires_grad=False).to(device).float()
		
		# if k not in self.padding_dict:
		# 	print(k)
		# 	self.padding_dict.update({
		# 		k:torch.zeros((b, c, h, w), requires_grad=False).to(device).float()
		# 	})
		return padding

	def forward(self, im0, im1, events, interp_ratio):

		pre_conv = self.pre_conv(torch.cat((im0, events, im1), 1))*self.ct

		events_split_list_ = pre_conv.split(self.cur_step, 1)

		paddings = self.get_padding(self.n, self.step-self.cur_step, self.h, self.w, im0.device)
		events_split_list = [torch.cat((et, paddings), 1) for et in events_split_list_]
		events_stack = torch.cat(events_split_list, 0)

		e_half = self.down0(events_stack)
	
		e_quater = self.down1(e_half)
		
		e_out = torch.stack(e_quater.split(self.n, 0), 1)
		return e_out, events_stack
