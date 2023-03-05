
from turtle import back
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet import BasePointNet
from .pointnet2 import BasePointNet2
from .dgcnn import BaseDgcnn
from .instance_counter import InstanceCounter               # : add for inference ins counter (MSE loss)


class Jsis(nn.Module):
    def __init__(self, backbone_type="dgcnn", addon=None):
        super(Jsis, self).__init__()

        self.num_classes = 2
        self.embedding_size = 32
        self.addon = addon

        if backbone_type == "pointnet":
            self.backbone = BasePointNet()
        elif backbone_type == "pointnet2":
            self.backbone = BasePointNet2()
        elif backbone_type == "dgcnn":
            self.backbone = BaseDgcnn()

        if self.addon is "counter":                         # : add for inference ins counter (MSE loss)
            self.counter = InstanceCounter()

        # self.fc1 = nn.Conv1d(128, self.num_classes, 1)
        # self.fc2 = nn.Conv1d(128, self.embedding_size, 1)


        self.emb_dims = 1024

        self.fc_1 = nn.Conv1d(self.emb_dims, 512, 1)
        self.fc_2 = nn.Conv1d(512, 256, 1)
        self.fc_3 = nn.Conv1d(256, self.num_classes, 1)

        self.fc_4 = nn.Conv1d(self.emb_dims, 512, 1)
        self.fc_5 = nn.Conv1d(512, 256, 1)
        self.fc_6 = nn.Conv1d(256, self.embedding_size, 1)



    def forward(self, x):
        x = self.backbone(x)
        # logits = self.fc1(x)
        # logits = logits.transpose(2, 1)
        # logits = torch.log_softmax(logits, dim=-1)
        # embedded = self.fc2(x)
        # embedded = embedded.transpose(2, 1)

        logits = self.fc_1(x)
        logits = self.fc_2(logits)
        logits = self.fc_3(logits)
        logits = logits.transpose(2, 1)
        logits = torch.log_softmax(logits, dim=-1)

        embedded = self.fc_4(x)
        embedded = self.fc_5(embedded)
        embedded = self.fc_6(embedded)

        embedded = embedded.transpose(2, 1)

        if self.addon is "counter":
            instance_count = self.counter(x)

            return x, logits, embedded, instance_count

        return x, logits, embedded

