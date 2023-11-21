import torch
import torch.nn as nn
import torch.nn.functional as F 

from .pointnet import PointNet
from .dgcnn import Dgcnn
from .pointmlp import PointMLPForUOP

class UOPNet(nn.Module):
    def __init__(self, backbone_type="dgcnn"):
        super(UOPNet, self).__init__()

        self.backbone_type = backbone_type

        if self.backbone_type == "pointnet":
            self.backbone = PointNet()

        elif self.backbone_type == "dgcnn":
            self.backbone = Dgcnn()

        elif self.backbone_type == "pointmlp":
            self.backbone = PointMLPForUOP()

        else:
            raise("Backbone type should be in [pointnet, dgcnn, pointmlp]. Baseline UOP model uses dgcnn")
        
        
        if self.backbone_type == "pointmlp":
            # semantic branch to predict a point stable or not
            self.sb_1 = nn.Conv1d(128, 128, 1)
            self.sb_2 = nn.Conv1d(128, 128, 1)
            self.sb_3 = nn.Conv1d(128, 2, 1) # num classes = 2  (stable, unstable)

            # instance branch to embed instance features for points
            self.ib_1 = nn.Conv1d(128, 128, 1)
            self.ib_2 = nn.Conv1d(128, 128, 1)
            self.ib_3 = nn.Conv1d(128, 32, 1) # embedding size = 32
        else:
            # semantic branch to predict a point stable or not
            self.sb_1 = nn.Conv1d(1024, 512, 1)
            self.sb_2 = nn.Conv1d(512, 256, 1)
            self.sb_3 = nn.Conv1d(256, 2, 1) # num classes = 2  (stable, unstable)

            # instance branch to embed instance features for points
            self.ib_1 = nn.Conv1d(1024, 512, 1)
            self.ib_2 = nn.Conv1d(512, 256, 1)
            self.ib_3 = nn.Conv1d(256, 32, 1) # embedding size = 32

    def forward(self, x):
        x = self.backbone(x)

        if self.backbone_type == "pointmlp":
            x = x.transpose(2, 1)

        out = self.sb_1(x)
        out = self.sb_2(out)
        out = self.sb_3(out)
        out = out.transpose(2, 1)
        logits = torch.log_softmax(out, dim=-1)

        emb = self.ib_1(x)
        emb = self.ib_2(emb)
        emb = self.ib_3(emb)
        emb = emb.transpose(2, 1)

        if self.backbone_type == "pointmlp":
            x = x.transpose(2, 1)


        return x, logits, emb
        
