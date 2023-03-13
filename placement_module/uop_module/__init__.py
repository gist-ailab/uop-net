import os
import torch
from .model import Jsis


class Config():
    def __init__(self, cfg):
        self.config = cfg


def load_model(partial=True, ckp=None, addon=None):
    model = Jsis(backbone_type="dgcnn", addon=addon)
    
    if ckp is None:
        if partial:
            ckp = 'partial_best'
        else:
            ckp = 'whole_best'
        ckp = os.path.join(os.path.dirname( os.path.abspath(__file__) ), "{}.pth".format(ckp))
    else:
        pass
    print('>>> Loading model from {}....'.format(ckp))
    model.load_state_dict(torch.load(ckp))
    model.cuda()
    model.eval()
    
    return model