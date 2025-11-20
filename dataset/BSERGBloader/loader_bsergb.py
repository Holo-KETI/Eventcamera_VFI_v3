import torch
import numpy as np
import os
from tools.registery import DATASET_REGISTRY
from dataset.BaseLoaders.baseloader import BaseLoader
import random
from numba import jit



indexing_skip_ind = {
    'basket_09':['000031.npz', '000032.npz', '000033.npz', '000034.npz'],
    'may29_rooftop_handheld_02':['000017.npz', '000070.npz'],
    'may29_rooftop_handheld_03':['000306.npz'],
    'may29_rooftop_handheld_05':['000121.npz'],
}


@jit(nopython=True)
def sample_events_to_grid(voxel_channels, h, w, x, y, t, p, hs, ws):

    x = (x-ws) / (19968 * w / h) * (w - 1)
    y = (y-hs) / 19968 * (h - 1)
  
    voxel = np.zeros((voxel_channels, h, w), dtype=np.float32)
    if len(t) == 0:
        return voxel
    t_start = t[0]
    t_end = t[-1]
    t_step = (t_end - t_start + 1) / voxel_channels
    for d in range(len(x)):
        d_x, d_y, d_t, d_p = x[d], y[d], t[d], p[d]
        d_x_low, d_y_low = int(d_x), int(d_y)
        d_t = d_t - t_start

        x_weight = d_x - d_x_low
        y_weight = d_y - d_y_low
        ind = int(d_t // t_step)
        pv = d_p * 2 - 1
        if d_y_low < h and d_x_low < w:
            voxel[ind, d_y_low, d_x_low] += (1 - x_weight) * (1 - y_weight) * pv
        if d_y_low + 1 < h and d_x_low < w:
            voxel[ind, d_y_low + 1, d_x_low] += (1 - x_weight) * y_weight * pv
        if d_x_low + 1 < w and d_y_low < h:
            voxel[ind, d_y_low, d_x_low + 1] += (1 - y_weight) * x_weight * pv
        if d_y_low + 1 < h and d_x_low + 1 < w:
            voxel[ind, d_y_low + 1, d_x_low + 1] += x_weight * y_weight * pv
    return voxel


@DATASET_REGISTRY.register()
class loader_bsergb(BaseLoader):
    def __init__(self, para, training=True):
        key = 'validation_config'
        self.real_interp = 5
        self.indexes = [0,1,2,3,4,5,6,7,8,9]

        super().__init__(para, training)
        self.norm_voxel = True
        self.sub_div = 24
        self.sample_t = [1,2,3,4]
        self.left_weight = [0.8000,0.6000,0.4000,0.2000]
        self.data_num = 10
        self.rgb_sampling_ratio =1
        
    
    def samples_indexing(self):
        self.samples_list = []
        for k in self.data_paths.keys():
            rgb_path, evs_path = self.data_paths[k]
            evs_len = len(evs_path)
            rgb_path = rgb_path[:evs_len]
            indexes = list(range(0, len(rgb_path),
                                 self.rgb_sampling_ratio))
            if k in indexing_skip_ind:
                skip_events = indexing_skip_ind[k]
            else:
                skip_events = []
            skip_sample = False
            for i_ind in range(0, len(indexes) - self.real_interp, 1 if self.training_flag else self.real_interp):
                rgb_sample = [rgb_path[sind] for sind in indexes[i_ind:i_ind + self.real_interp + 1]]
                evs_sample = evs_path[indexes[i_ind]:indexes[i_ind + self.real_interp]]
                rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
                for epath in evs_sample:
                    ename = os.path.split(epath)[-1]
                    if ename in skip_events:
                        print(f"Skip sample: {k:50}\t {ename}")
                        skip_sample = True
                        break
                if not skip_sample:
                    self.samples_list.append([k, rgb_name, rgb_sample, evs_sample])
                skip_sample = False

        return
    # def samples_indexing(self):
    #     self.samples_list = []
    #     # 이벤트 합친 version.
    #     for k in self.data_paths.keys():
    #         rgb_path, evs_path = self.data_paths[k]
    #         for i_ind in range(0, len(self.indexes)-1):
    #             rgb_sample = [rgb_path[i_ind],rgb_path[i_ind+1]]
    #             # evs_sample = evs_path[i_ind*5:i_ind*5 + self.real_interp]
    #             # print(evs_path)
    #             evs_sample = evs_path[i_ind:i_ind+1]
    #             rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
    #             for epath in evs_sample:
    #                 ename = os.path.split(epath)[-1]
    #             self.samples_list.append([k, rgb_name, rgb_sample, evs_sample])

    #     return
        
    def events_reader(self, events_path, h, w, hs, ws):
        evs_data = [np.load(ep) for ep in events_path]
        evs_voxels = []
       
        for ed in evs_data:
            evs_voxels.append(sample_events_to_grid(self.sub_div, h, w, np.float32(ed['x']),
                                                    np.float32(ed['y']), np.float32(ed['timestamp']), np.float32(ed['polarity']),
                                                    hs, ws))
        return torch.tensor(np.concatenate(evs_voxels, 0))

    def data_loading(self, paths):
        
        folder_name, rgb_name, rgb_sample, evs_sample = paths

        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        h, w = im0.shape[1:]
        

        events = self.events_reader(evs_sample, h, w, 0, 0)
        return folder_name, rgb_name, im0, im1, events
    
    def __getitem__(self, item):
        item_content = self.samples_list[item]
        folder_name, rgb_name, im0, im1, events = self.data_loading(item_content)
        h, w = im0.shape[1:]
        hn, wn = (h//32-1)*32, (w//32-1)*32
        hleft = (h-hn)//2
        wleft = (w-wn)//2
        im0, im1, events = im0[:, hleft:hleft+hn, wleft:wleft+wn], im1[:, hleft:hleft+hn, wleft:wleft+wn], events[:, hleft:hleft+hn, wleft:wleft+wn]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        rgb_name = [rgb_name[0]]  + [rgb_name[-1]]
        # data_back = {
        #     'folder': folder_name,
        #     'rgb_name': [rgb_name[0]]  + [rgb_name[-1]],
        #     'im0': im0,
        #     'im1': im1,
        #     'events': events,
        #     't_list': self.sample_t,
        #     'left_weight': self.left_weight,
        #     'interp_ratio':self.interp_ratio
        # }
        return folder_name, rgb_name, im0, im1, events, self.sample_t, self.left_weight, self.interp_ratio
