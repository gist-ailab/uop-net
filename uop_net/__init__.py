from uop_net.module.model.uopnet import UOPNet
import torch
import os

def load_model(partial=True, backbone='dgcnn', weight=None):
    model = UOPNet(backbone_type=backbone)
    
    if weight is None:
        if partial:
            weight = 'partial_best'
        else:
            weight = 'whole_best'
        weight = os.path.join(os.path.dirname( os.path.abspath(__file__) ), "{}.pth".format(weight))
        
    else:
        pass
        
    print('>>> Loading model from {}....'.format(weight))
    model.load_state_dict(torch.load(weight))
    model.cuda()
    model.eval()
    
    return model