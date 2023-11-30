import torch
from torch.utils.data.dataset import random_split

from data.uop_sim import UOPSim, MergeDataset


def load_data(args):
    """
    root_1 : Custom ShapeNet root
    root_2 : Custom 3DNet root
    """
    root_1 = args.shapenet_root
    root_2 = args.threednet_root

    root_list = [root_1, root_2]
    dataset_list = []
    for root in root_list:
        dataset_list.append(UOPSim(root,
                            sampling='random',
                            partial=False,  #TODO
                            projection=False))
    dataset = MergeDataset(dataset_list)
    data_length  = dataset.dataset_size
    tr_data, val_data = random_split(dataset, [round(data_length * 0.8), round(data_length * 0.2)], generator=torch.Generator().manual_seed(777))

    return tr_data, val_data

