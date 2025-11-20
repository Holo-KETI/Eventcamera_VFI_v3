import torch
import sys
import os
sys.path.append(os.getcwd())
from thop import profile
from models.Expv8_large.runExpv8_large import Expv8_large
# from params.GOPRO_release.params_trainOurs_mix import trainGOPRO_Ours
from params.bsergb.params_traintest_x5AdamwithLPIPS_vali import traintest_BSERGB_x5AdamwithLPIPS_vali
from easydict import EasyDict as ED
import time

import torch
import sys
import os
sys.path.append(os.getcwd())
import torch
from torch import nn
from torch.nn import functional as F

from dataset.BSERGBloader.loader_bsergb import loader_bsergb
from tools.registery import DATASET_REGISTRY, LOSS_REGISTRY, PARAM_REGISTRY
from torch.utils.data import DataLoader

from tools.file_path_index import parse_path_common
from decoder_class import FlowDecoder
from masknet_class import MaskNet
from image_class import ImageEncoder
from event_class import EventsEncoder
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage

import random
def safe_clone(tensor):
    return tensor.clone() if tensor is not None else None

def validation(testLoader, image_encoder,event_encoder ,decoder_0,decoder_1, masknet):

    image_encoder.eval() 
    event_encoder.eval()
    decoder_0.eval()
    decoder_1.eval()
    masknet.eval()

    with torch.no_grad():
        for _, testdata in enumerate(testLoader):
                ## data
    
            with torch.no_grad():
                left_frame, right_frame, events = testdata[2].cuda(), \
                    testdata[3].cuda(), testdata[4].cuda()
                interp_ratio = 10

            e_out, e_out_large = event_encoder(left_frame, right_frame, events, interp_ratio)
            im0_feat, im1_feat, im0_feat_full, im1_feat_full = image_encoder(left_frame, right_frame)
            
            n, t, c, h, w = e_out.shape
            
            
            
            forward_ims = []
            backward_ims = []
            forward_cur_flow = torch.zeros((n, 2, h, w)).to("cuda")
            backward_cur_flow = torch.zeros((n, 2, h, w)).to("cuda")
            forward_cur_feat = im0_feat
            backward_cur_feat = im1_feat
            forward_Delta_flow_list, backward_Delta_flow_list = [], []
            forward_econv, backward_econv = im0_feat, im1_feat
            forward_flow_list = []
            backward_flow_list = []


            
            
            for i in [0, 1, 2, 3, 4, 5, 6,7]:
                forward_cur_feat, forward_cur_flow, forward_cur_flow_large, forward_econv = decoder_0(e_out[:,i], im0_feat, forward_cur_flow,forward_cur_feat,forward_econv)
                backward_cur_feat, backward_cur_flow, backward_cur_flow_large, backward_econv = decoder_1(e_out[:,t - i - 1], im1_feat, backward_cur_flow,backward_cur_feat,backward_econv)
                
                
                forward_ims.append(decoder_0.backwarp(left_frame, safe_clone(forward_cur_flow_large), 576, 928))
                backward_ims.insert(0, decoder_0.backwarp(right_frame, safe_clone(backward_cur_flow_large), 576, 928))
                forward_Delta_flow_list.append(safe_clone(forward_econv))
                backward_Delta_flow_list.insert(0, safe_clone(backward_econv))
                forward_flow_list.append(safe_clone(forward_cur_flow_large))
                backward_flow_list.insert(0, safe_clone(backward_cur_flow_large))
            

        
            
            resout = []
            fuseout = []
            tmask = torch.ones((1, 1, 576, 928)).to("cuda")

            
            end_time= [1,3,5,7]
            interp_ratio = 10
            for t_ in end_time:
                res, m, fo = masknet(left_frame,
                                    right_frame,
                                    forward_ims[t_],
                                    backward_ims[t_ - end_time[0]],
                                    (interp_ratio - t_ - 1) / interp_ratio * tmask,
                                    forward_flow_list[t_],
                                    backward_flow_list[t_ - end_time[0]],
                                    forward_Delta_flow_list[t_],
                                    backward_Delta_flow_list[t_ - end_time[0]],
                                    im0_feat_full,
                                    im1_feat_full
                                    )
                resout.append(res)
                fuseout.append(fo)

       
    return  torch.stack(resout, 1)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 멀티 GPU 환경 대비
        # 비결정적 알고리즘 사용 비활성화
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False


if __name__=="__main__":
    # set_seed(42)
    args = ED()
    args.model_name = 'Expv8_large'
    args.extension = ''
    args.clear_previous = None
    args.model_pretrained = None
    args.calc_flops = True
    # args.param_name = 'traintest_BSERGB_x4AdamwithLPIPS'

    params = traintest_BSERGB_x5AdamwithLPIPS_vali(args)
    # params = args.param_name
    # records = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{args.model_name}_flops_and_macs.txt'), 'a+')
    # params.training_config.crop_size = 64

    params.events_channel = 120
    params.validation_config.interp_ratio = 10
    
    # params.real_interp = 2
    params.real_interp = 5
    params.save_flow = False
    params.save_images = False

    params.debug = False
    params.validation_config.data_paths = parse_path_common('/home/sieun/test_final/Eventcamera_VFI_v2/dataset/test_data', '/home/sieun/test_final/Eventcamera_VFI_v2/dataset/test_data',bsergb=True)
    

	
    testDataset = DATASET_REGISTRY.get(params.validation_config.dataloader)(params, training=False)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=1)
	
    ## image encoder
    image_encoder = ImageEncoder(img_ch=3, base_channel=32)  # 반드시 같은 구조
    image_encoder.load_state_dict(torch.load("/home/sieun/test_final/Eventcamera_VFI_v2/51/image_encoder_51.pt")['state_dict'])
    image_encoder.cuda()
    
    ## event encoder
    event_encoder = EventsEncoder(120, 6 , 32,num_decoder=12)
    event_encoder.load_state_dict(torch.load("/home/sieun/test_final/Eventcamera_VFI_v2/51/event_encoder_51.pt")['state_dict'])
    event_encoder.cuda()

    ## decoder
    decoder_0 = FlowDecoder(32 * 12, 32 * 4, 32 * 4)
    decoder_1 = FlowDecoder(32 * 12, 32 * 4, 32 * 4)
    
    decoder_0.load_state_dict(torch.load("/home/sieun/test_final/Eventcamera_VFI_v2/51/decoder_0_51.pt")['state_dict'])
    decoder_0.cuda()

    decoder_1.load_state_dict(torch.load("/home/sieun/test_final/Eventcamera_VFI_v2/51/decoder_1_51.pt")['state_dict'])
    decoder_1.cuda()
    
    ## masknet
    masknet = MaskNet(5 + 6 * 2, 32 * 4)
    masknet.load_state_dict(torch.load("/home/sieun/test_final/Eventcamera_VFI_v2/51/masknet_51.pt")['state_dict'])
    masknet.cuda()

    recon = validation(testLoader, image_encoder,event_encoder ,decoder_0,decoder_1, masknet)
    # breakpoint()

    
    ## 이미지 저장해서 보기
    for n in range(recon.shape[1]):
        epoch = 51
        toim = ToPILImage()
        os.makedirs(os.path.join("./", str(epoch)), exist_ok=True)
        img_tensor =recon[0, n].detach().cpu().clamp(0, 1)
        # img_tensor = img_tensor[[2, 1, 0], :, :]
        toim(img_tensor).save(os.path.join("./", str(epoch), f"test_{n}_res.png"))
            

		

		
