from flow_vis import flow_to_color
import torch
from torch.nn import Module
import losses
from tools.registery import LOSS_REGISTRY
from copy import deepcopy
import time
from torchvision.transforms import ToTensor, ToPILImage
import os
import numpy as np

import os
from flow_vis import flow_to_color
mkdir = lambda x:os.makedirs(x, exist_ok=True)


class BaseModel(Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.train_im_path = params.paths.save.train_im_path
        # self.val_im_path = params.paths.save.val_im_path

        self.val_im_path = os.path.join(params.paths.save.val_im_path, "images")
        self.val_flow_save = os.path.join(params.paths.save.val_im_path, "flow")
        self.val_flowvis_save = os.path.join(params.paths.save.val_im_path, "flow_vis")
        mkdir(self.val_im_path)
        mkdir(self.val_flow_save)
        mkdir(self.val_flowvis_save)
        self.record_txt = params.paths.save.record_txt
        self.val_record_txt = os.path.join(self.val_im_path, 'detailed_records')

        self.training_metrics = {}
        self.validation_metrics = {}
        self.train_print_freq = params.training_config.train_stats.print_freq
        self.train_im_save = params.training_config.train_stats.save_im_ep
        self.val_eval = params.validation_config.weights_save_freq
        self.val_im_save = params.validation_config.val_imsave_epochs
        self.interp_num = params.training_config.interp_ratio - 1
        self.toim = ToPILImage()


    
        self.params_training = None
        self.debug=params.debug
        self.save_flow = params.save_flow
        self.save_images = params.save_images


    def write_log(self, logcont):
        with open(self.record_txt, 'a+') as f:
            f.write(logcont)


    def net_training(self, data_in, optim, epoch, step):
        pass

    def validation(self, data_in, epoch):
        pass

    def forward(self, *args, **kwargs):
        pass

   














