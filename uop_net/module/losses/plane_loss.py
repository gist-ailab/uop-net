import torch
import torch.nn as nn
import numpy as np

class PlaneLoss(nn.Module):
    def __init__(self, delta_d, delta_v,
                 alpha=1.1, beta=1.1, gamma=0.001, 
                 reduction='mean'):
        
        super(PlaneLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Set delta_d > 2 * delta_v
        self.delta_d = delta_d
        self.delta_v = delta_v

    def forward(self, embedded, masks, size):
        centroids = self._centroids(embedded, masks, size)
        
        L_v = self._variance(embedded, masks, centroids, size)
        L_d = self._distance(centroids, size)
        L_r = self._regularization(centroids, size)

        loss = self.alpha * L_v + self.beta * L_d + self.gamma * L_r
        return loss

    def _centroids(self, embedded, masks, size):
        batch_size = embedded.size(0)                       # 8
        embedding_size = embedded.size(2)                   # 32 + 3
        masks = masks.long()
        K = masks.size(2)
        x = embedded.unsqueeze(2).expand(-1, -1, K, -1)     # : ([8, 2048, 16, 32])
        masks = masks.unsqueeze(3)                          # : ([8, 2048, 16, 1])
        
        
        x = x * masks                                       # : torch.Size([8, 2048, 16, 32])
        centroids = []
        for i in range(batch_size):
            n = size[i]
            mu = x[i,:,:n].sum(0) / (masks[i,:,:n].sum(0) + 0.001) # n, 32 / n, 1(points num of each instance)
            mu_debag = np.argwhere(torch.isnan(mu.cpu()) == True)
            if len(mu_debag[0]) != 0:
                print('here-mu_debag')
            mu.cuda()
            
            if K > n:
                m = int(K - n)
                filled = torch.zeros(m, embedding_size) # 16-n, 32
                filled = filled.to(embedded.device) 
                mu = torch.cat([mu, filled], dim=0) # 16, 32
            centroids.append(mu)
        centroids = torch.stack(centroids)                  # len : 8
        return centroids                                    # size : (16, 10, 32)

    def _variance(self, embedded, masks, centroids, size):
        batch_size = embedded.size(0)
        num_points = embedded.size(1)
        
        masks = masks.long()
        K = masks.size(2)
        # Convert input into the same size
        mu = centroids.unsqueeze(1).expand(-1, num_points, -1, -1)
        x = embedded.unsqueeze(2).expand(-1, -1, K, -1)
        # Calculate intra pull force

        var0 = torch.norm(x - mu, 2, dim=3)
        var1 = torch.clamp(var0 - self.delta_v, min=0.0) ** 2
        var = var1 * masks
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            loss += (torch.sum(var[i,:,:n]) / (torch.sum(masks[i,:,:n]) + 0.001))
        loss /= batch_size
        return loss


    def _distance(self, centroids, size):
        batch_size = centroids.size(0)
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            if n <= 1: continue
            mu = centroids[i, :n, :]
            mu_a = mu.unsqueeze(1).expand(-1, n, -1)   #(2, 3, 32)
            mu_b = mu_a.permute(1, 0, 2)                #(3, 2, 32)

            diff = mu_a - mu_b
            norm = torch.norm(diff, 2, dim=2)
            # norm = torch.norm(diff, 2, dim=1)
            margin = 2 * self.delta_d * (1.0 - torch.eye(n))
            margin = margin.to(centroids.device)
            distance = torch.sum(torch.clamp(margin - norm, min=0.0) ** 2)
            distance /= float(n * (n - 1) + 0.0001)
            loss += distance
        loss /= batch_size
        return loss

    def _regularization(self, centroids, size):
        batch_size = centroids.size(0)
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            mu = centroids[i, :n, :]
            norm = torch.norm(mu, 2, dim=1) + 0.0001
            loss += torch.mean(norm)
        loss /= batch_size
        return loss

if __name__ == '__main__':
    sample = torch.randn(16, 2048, 3)
    embedded = torch.randn(16, 2048, 32)
    masks = torch.randn(16, 2048, 1)
    size = (16, 5)

    delta_d, delta_v = 1.5, 0.5
    criterion = DiscriminativeLoss(delta_d, delta_v,)
    
    loss = criterion(embedded, masks, size)
    print(loss)