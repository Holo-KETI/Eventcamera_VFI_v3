from models.BaseModel import  BaseModel
from .archs import define_network

import torch
from tools.registery import MODEL_REGISTRY
import os




@MODEL_REGISTRY.register()
class Expv8_large(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.net = define_network(params.model_config.define_model)
        self.net.debug = self.debug
        self.grad_cache = {}
        ## 수정
        self.interp_ratio = 10
        self.real_interp = 5
        self.end_tlist = [1,3,5,7]
        self.scalar = 4 #self.real_interp - 1



    def net_validation(self, data_in, epoch):
        self.eval()
        with torch.no_grad():
            left_frame, right_frame, events = data_in[2].cuda(), \
                data_in[3].cuda(), data_in[4].cuda()
          
            recon = self.forward(left_frame,
                                 right_frame,
                                 events)[0]

            recon_min = torch.min(recon)
            recon_max = torch.max(recon)
            recon = (recon - recon_min) / (recon_max - recon_min + 1e-8)


        
        ## 이미지 저장해서 보기
        for n in range(recon.shape[1]):
            os.makedirs(os.path.join(self.val_im_path, str(epoch)), exist_ok=True)
            rgb_name = data_in[1]
   
            folder = os.path.split(data_in[0][0])[-1] 
            self.toim(recon[0, n].detach().cpu().clamp(0, 1)).save(os.path.join(self.val_im_path, str(epoch), f"{folder}_{rgb_name[0][0]}_{n}_res.png"))
           
    def forward(self, left_frame, right_frame, events):    

        res = self.net(torch.cat((left_frame, right_frame), 1), events, self.interp_ratio, self. end_tlist)
        
        return res