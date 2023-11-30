import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from module.model.uopnet import UOPNet

from module.losses.discriminative_loss import DiscriminativeLoss
from module.losses.nll_loss import NLLLoss

from data.load_data import load_data

from warmup_scheduler import GradualWarmupScheduler



def train(args):
    """
    get arguments from parser for training UOP-Net
    """
    
    backbone_type = args.backbone_type
    ckp_path = args.ckp_path

    stab_scale = args.stab_scale
    plane_scale = args.plane_scale
    delta_d = args.delta_d
    delta_v = args.delta_v
    

    """
    set dataset
    """
    tr_data, val_data = load_data(args)
    tr_loader = DataLoader(tr_data,
                           batch_size=args.batch_size,
                           num_workers=args.num_workers,
                           shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)

    lr = args.lr

    optimizer = optim.Adam(model_params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)




    """
    load UOP-Net
    """
    model = UOPNet(backbone_type=backbone_type)
    model_params = model.parameters()

    model = model.to(device)

    """
    load Losses
    """
    criterion ={}
    criterion["stab"] = NLLLoss()
    criterion["plane"] = DiscriminativeLoss(delta_d=delta_d, delta_v=delta_v)

    criterion["stab"], criterion["plane"] = criterion["stab"].to(device), criterion["plane"].to(device)    

    best_loss = np.Inf
    
    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        scheduler_warmup.step()

        for i, data in enumerate(tr_loader): 
            points = data["points"]
            labels = data["sem_labels"]
            masks = data["ins_labels"]
            size = data["size"]
            
            points, labels, masks, size = points.to(device), labels.to(device), masks.to(device), size.to(device)

            loss = 0.0

            feat, logits, embedded = model(points)
            loss_stab = criterion["stab"](logits, labels)
            loss_plane = criterion["plane"](embedded, masks, size)

            loss += loss_stab * stab_scale
            loss += loss_plane * plane_scale
            print(">>>>>epoch : {} || nll loss : {} || dis loss : {} || total loss : {}".format(epoch, loss_stab, loss_d, loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # for validation

        if epoch % 10 == 0:
            overall_val_loss = []
            model.eval()
            with torch.no_grad():
                for j, data in enumerate(val_loader):
                    points = data["points"]
                    labels = data["sem_labels"]
                    masks = data["ins_labels"]
                    size = data["size"]

                    points, labels, masks, size = points.to(device), labels.to(device), masks.to(device), size.to(device)

                    val_loss = 0.0
                    feat, logits, embedded = model(points)
                    val_loss_n = criterion["nll"](logits, labels)
                    val_loss_d = criterion["dis"](embedded, masks, size)

                    val_loss += val_loss_n
                    val_loss += val_loss_d
                    overall_val_loss.append(val_loss)
                    print(">>>>>epoch : {} || val nll loss : {} || val dis loss : {} || total val loss : {}".format(epoch, val_loss_n, val_loss_d, val_loss))
                
            if (sum(overall_val_loss)/(len(overall_val_loss))) < best_loss:
                ckp_dir = os.path.join(ckp_path, str(backbone_type)+"_"+str(stab_scale)+"_"+str(plane_scale))
                os.makedirs(ckp_dir, exist_ok=True)

                model_state_dict = model.state_dict()

                state = {'model_state_dict' : model_state_dict}

                save_path = str(ckp_dir) + '/ep%03d_model.pth' % epoch
                torch.save(state, save_path)

                print('>>> Saving model to {}...'.format(save_path))




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--points_shape", type=str, default="partial",
                        help="points shape should be in [whole, partial]")
    parser.add_argument("--backbone_type", type=str, default="dgcnn",
                        help="backbone type should be in [pointnet, dgcnn]")
    parser.add_argument("--shapenet_root", type=str)
    parser.add_argument("--threednet_root", type=str)
    parser.add_argument("--ckp_path", type=str, default=None)
    parser.add_argument("--stab_scale", type=float, default=10.0)
    parser.add_argument("--plane_scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--delta_d", type=float, default=1.5, help="delta_d for plane loss")
    parser.add_argument("--delta_v", type=float, default=0.5, help="delta_v for plane loss")
    parser.add_argument("--epochs", type=int, default=1000, help="epochs for training")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="num workers for training")
    

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train(args, device)