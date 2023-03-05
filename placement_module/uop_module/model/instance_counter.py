import torch
import torch.nn as nn
import torch.nn.functional as F

class InstanceCounter(nn.Module):
    def __init__(self):
        super(InstanceCounter, self).__init__()
        """
        in_channel = features output size
        """
        # self.opt = opt
        # self.input_channels = self.opt.config['base']['model']['dgcnn']['emb_dims']
        self.input_channels = 1024              # : add for inference ins counter (MSE loss)
    
        self.counter_1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.counter_2 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(), 
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU() 
        )
        self.linear = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x is features of encoder
        """
        batch_size = x.shape[0]
        num_points = x.shape[2]

        x = F.max_pool1d(x, num_points).squeeze(2)          #[16, 1024]
        x = x.view(-1, 1024, 1).repeat(1, 1, num_points)    #[16, 1024, 2048]
        x = self.counter_1(x)                               #[16, 256, 2048]
        x = F.max_pool1d(x, num_points).squeeze(2)          #[16, 256]
        x = x.view(-1, 256, 1).repeat(1, 1, num_points)     #[16, 256, 2048]
        x = self.counter_2(x)                               #[16, 64, 2048]
        
        x = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1) #[16, 64]
        x = self.linear(x)                                   #[16, 1]
        x = self.sigmoid(x)                                  #[16, 1]

        return x



