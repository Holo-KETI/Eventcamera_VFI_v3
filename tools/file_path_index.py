import os
from natsort import natsorted
try:
    from dataset.RC_4816.dataset_dict import EVSneg3
except:
    print("EVSneg3 do not load")
    pass


def parse_path(folder, fdepth):
    file_dict = {}
    if fdepth == 1:
        file_dict.update({
            os.path.split(folder)[-1]:[os.path.join(folder, file) for file in natsorted(os.listdir(folder))]
        })
    else:
        for subfolder in os.listdir(folder):
            sfolder = os.path.join(folder, subfolder)
            file_dict.update({
                subfolder:[os.path.join(sfolder, file) for file in natsorted(os.listdir(sfolder))]
            })
    return file_dict


def parse_path_common(folder0, folder1, fdepth=2, cbmnet=False, hsergb=False, bsergb=False, RC=False):
    file_dict = {}
    
    if bsergb:
        if folder1 is None:
            folder1 = folder0
        folders = os.listdir(folder0)
        for folder in natsorted(folders):
            try:
                file_dict.update({
                    folder : [[os.path.join(folder0, folder, 'images', pi) for pi in natsorted(os.listdir(os.path.join(folder0, folder, 'images')))],
                           [os.path.join(folder1, folder, 'events', pe) for pe in natsorted(os.listdir(os.path.join(folder1, folder, 'events')))]]
                })
            except:
                pass
   
    return file_dict
