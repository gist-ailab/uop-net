import torch
import torch.nn as nn
import torch.nn.functional as F

'''backbone network for uop'''
class STN3D(nn.Module):
    def __init__(self, input_channels):
        super(STN3D, self).__init__()
        self.input_channels = input_channels
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_channels * input_channels)
        )

    def forward(self, x):
        
        
        batch_size = x.shape[0]
        num_points = x.shape[2]

        x = self.mlp1(x)
        x = F.max_pool1d(x, num_points).squeeze(2)
        x = self.mlp2(x)
        I = torch.eye(self.input_channels).view(-1).to(x.device)
        x = x + I
        x = x.view(-1, self.input_channels, self.input_channels)
        return x


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
    
        self.input_channels = 3

        self.stn1 = STN3D(self.input_channels)
        self.stn2 = STN3D(64)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(1088, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

    def forward(self, x):
        
        num_points = x.shape[1]

        x = x.transpose(2, 1)
        T = self.stn1(x)
        x = torch.bmm(T, x)
        x = self.mlp1(x)
        T = self.stn2(x)
        f = torch.bmm(T, x)
        x = self.mlp2(x)
        x = F.max_pool1d(x, num_points).squeeze(2)
        x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
        x = torch.cat([x, f], 1)
        x = self.mlp3(x)
        return x

class BasePointNet(nn.Module):
    def __init__(self, opt):
        super(BasePointNet, self).__init__()
        self.opt = opt        
        if self.opt.config['base']['train']['camera_info']:
            if self.opt.config['base']['train']['normals']:
                self.input_channels = 7
            else:
                self.input_channels = 4
        else:
            if self.opt.config['base']['train']['normals']:
                self.input_channels = 6
            else:
                self.input_channels = 3

        self.stn1 = STN3D(self.input_channels)
        self.stn2 = STN3D(64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

    def forward(self, x):

        x = torch.tensor(x, dtype=torch.float32)

        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x.transpose(2, 1) # transpose to apply 1D convolution
        T = self.stn1(x)
        x = torch.bmm(T, x)
        x = self.mlp1(x)
        T = self.stn2(x)
        f = torch.bmm(T, x)
        x = self.mlp2(f)
        x = F.max_pool1d(x, num_points).squeeze(2) # max pooling
        x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
        x = torch.cat([x, f], 1)
        x = self.mlp3(x) # should be Bx128xN here
        return x