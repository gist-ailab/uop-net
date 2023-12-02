import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data.dataset import random_split

from dataset.uopsim import UOPSIM



class MergeDataset(UOPSIM):
    def __init__(self, dataset_list):
        self.dataset_size = 0
    
        self.dataset_list = dataset_list
    
        self.size_list = []
        for dataset in dataset_list:
            self.dataset_size += dataset.dataset_size
            self.size_list.append(self.dataset_size)
    
        print(">>>Merge Dataset Size: {}".format(self.dataset_size))
    
    def __getitem__(self, i):
        prev = 0
        for idx, size in enumerate(self.size_list):
            if i < size:
                return self.dataset_list[idx][i-prev]
            prev = size



def load_data(args):
    """
    root_1 : Custom ShapeNet root
    root_2 : Custom 3DNet root
    """
    root_1 = args.shapenet_root
    root_2 = args.threednet_root

    if args.points_shape == "partial":
        is_partial = True
    elif args.points_shape == "whole":
        is_partial = False
    else:
        NotImplementedError


    root_list = [root_1, root_2]
    dataset_list = []
    for root in root_list:
        dataset_list.append(UOPSIM(root,
                            sampling='random',
                            partial=is_partial,  
                            ))
    dataset = MergeDataset(dataset_list)
    data_length  = dataset.dataset_size
    tr_data, val_data = random_split(dataset, [round(data_length * 0.8), round(data_length * 0.2)], generator=torch.Generator().manual_seed(777))

    return tr_data, val_data

