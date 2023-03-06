from module.model.dgcnn import BaseDgcnn

import torch
import torch.nn as nn
import torch.nn.functional as F

class UOPNet(nn.Module):
    def __init__(self, opt):
        super(UOPNet, self).__init__()
        self.opt = opt
        self.num_classes = opt.config['base']['train']['num_classes']
        self.embedding_size = opt.config['base']['train']['embedding_size']
    
        self.backbone = BaseDgcnn(self.opt)

        self.emb_dims = opt.config['base']['model']['dgcnn']['emb_dims']    
        
        self.fc_1 = nn.Conv1d(self.emb_dims, 512, 1)
        self.fc_2 = nn.Conv1d(512, 256, 1)
        self.fc_3 = nn.Conv1d(256, self.num_classes, 1)

        self.fc_4 = nn.Conv1d(self.emb_dims, 512, 1)
        self.fc_5 = nn.Conv1d(512, 256, 1)
        self.fc_6 = nn.Conv1d(256, self.embedding_size, 1)
    
    def forward(self, x):
            
        x = self.backbone(x)

        logits = self.fc_1(x)
        logits = self.fc_2(logits)
        logits = self.fc_3(logits)
        logits = logits.transpose(2, 1)
        logits = torch.log_softmax(logits, dim=-1)

        embedded = self.fc_4(x)
        embedded = self.fc_5(embedded)
        embedded = self.fc_6(embedded)

        embedded = embedded.transpose(2, 1)


        return x, logits, embedded