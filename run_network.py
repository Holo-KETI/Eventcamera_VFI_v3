from tools.registery import DATASET_REGISTRY, LOSS_REGISTRY, PARAM_REGISTRY
import params, losses, models, dataset
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tools.interface_deparse import keyword_parse
from tools.model_deparse import *
from tqdm import tqdm
import time
from models.Expv8_large.runExpv8_large import Expv8_large
from torchvision.transforms import ToTensor, ToPILImage


def main():
    print('In Network')
    params = keyword_parse()

    model, epoch, metric  = deparse_model(params)

    test_num_workers = 16
    testDataset = DATASET_REGISTRY.get('loader_bsergb')(params, training=False)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=test_num_workers)
    print('[Testing Samples Num]', len(testDataset))
    print("Num workers", test_num_workers)

    print('Do not train, start validation')

    for _, testdata in enumerate(testLoader):
 
        model.net_validation(testdata, epoch)

    return

if __name__ == '__main__':
    main()


