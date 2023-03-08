import os
import datetime
from re import U
from cv2 import split, transform
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
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler

from collections import defaultdict

from data.load_data import load_data
from utility.config import *
from utility import pointcloud_utils as pcu

from module.model.uop_net import UOPNet
from module.losses.plane_loss import PlaneLoss
from module.losses.stability_loss import StabilityLoss

def train(opt, device):
    
    tr_transform = transforms.Compose(
        [
            pcu.PointcloudRotate_batch(), 
            pcu.PointcloudRotatePerturbation_batch(),
            pcu.PointcloudJitter_batch()
        ]
    )
    
 
    tr_data, val_data = load_data(opt)
    tr_loader = DataLoader(tr_data,
                           batch_size=opt.config['base'],
                           num_workers=opt.config['base'],
                           shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=opt.config['base'],
                            num_workers=opt.config['base'],
                            shuffle=False)

    model = UOPNet(opt)
    model_params = model.parameters()

    optimizer = optim.Adam(model_params, lr=opt.config['base'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    loss_weight = None
    criterion = {}
    criterion['plane'] = PlaneLoss(delta_d=opt.config['base']['loss']['delta_d'],
                                                     delta_v=opt.config['base']['loss']['delta_v'])
    criterion['plane'].to(device)
    criterion['stab'] = StabilityLoss(loss_weight)
    criterion['stab'].to(device)


    best_loss = np.Inf
    start_epoch = 0
    total_epochs = opt.config['base']
    for epoch in range(start_epoch, total_epochs):
        
        if scheduler is not None:
            scheduler_warmup.step()
        
        scalars = defaultdict(list)

        batch_idx = 0
        for i, data in enumerate(tr_loader):
            model.train()
            
            points = data['points']
            points = tr_transform(points)

            points = points.to(device)
            labels = data['sem_labels'].to(device)
            masks = data['ins_labels'].to(device)
            size = data['size']

            loss = 0

            feat, logits, embedded = model(points)
            loss_s = criterion['stab'](logits, labels)
            loss_p = criterion['plane'](embedded, masks, size)

            loss += loss_s * 10 # 10 is a hyper-parameter for scaling loss scalar
            loss += loss_p * 1  # 1 is a hyper-parameter for scaling loss scalar

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scalars['loss'].append(loss)

            batch_idx += 1

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                v_batch_idx = 0
                overall_val_loss = []
                for j, data in enumerate(val_loader):
                    points = data['points'].to(device)
                    labels = data['sem_labels'].to(device)
                    masks = data['ins_labels'].to(device)
                    size = data['size']

                    val_loss = 0
                    feat, logits, embedded = model(points)
                    val_loss_s = criterion['stab'](logits, labels)
                    val_loss_p = criterion['plane'](embedded, masks, size)

                    val_loss += val_loss_s * 10
                    val_loss += val_loss_p * 1

                    overall_val_loss.append(val_loss)
            
            if (sum(overall_val_loss) / len(overall_val_loss)) < best_loss:
                ckp_path = os.path.join(ckp_dir, 'ckp')
                os.makedirs(ckp_path, exist_ok=True)

                model_state_dict = model.state_dict()
                
                state = {'model' : model,
                         'criterion' : criterion,
                         'save_epoch' : epoch,
                         'model_state_dict' : model_state_dict,
                         'optimizer' : optimizer.state_dict(),
                         'loss_value' : best_loss}
                save_path = str(ckp_path) + '/ep%03d_model.pth' % epoch
                torch.save(state, save_path)

                print(">>>>saving model and state to {}...".format(save_path))



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



