import os
import datetime
from re import U
from cv2 import split
import numpy as np
import argparse
import random
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append("uop_net")


from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from collections import defaultdict

from data.load_data import load_data
from utility.config import *
from utility.pointcloud_utils import *

from module.model.uop_net import UOPNet
from module.losses.plane_loss import PlaneLoss
from module.losses.stability_loss import StabilityLoss

def train(opt, device):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config', 
                        help='config file path')
    
    args = parser.parse_args()

    opt = config(args.config_path)
    opt.get_base_config()

    '''setting base path such as ckp_path'''
    save_dir = opt.config['base']['env']['save_dir']
    exp_name = opt.config['base']['env']['exp_name']
    exp_condition = opt.config['base']['env']['exp_condition']

    ckp_dir = os.path.join(save_dir, exp_name, exp_condition)
    os.makedirs(ckp_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(opt, device)



