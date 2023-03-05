import torch
import torch.nn as nn
import torch.nn.functional as F

from .pc_utils import *


class BasePointNet2(nn.Module):
    def __init__(self):
        super(BasePointNet2, self).__init__()
        
        self.input_channels = 3

        self.sa_1 = PointNetSetAbstractionMsg(512, [0.1, 0.2], [16, 32], self.input_channels, [[16, 16, 32], [32, 32, 64], [64, 64, 128]])
        self.sa_2 = PointNetSetAbstractionMsg(256, [0.2, 0.4], [64, 128], 32+64, [[128, 128, 256], [128, 196, 256]])
        self.sa_3 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [128, 256], 256+256, [[128, 128, 256], [128, 196, 256]])
        self.fp3 = PointNetFeaturePropagation(in_channel=256+256+256+256, mlp=[1120, 1024])
        self.fp2 = PointNetFeaturePropagation(in_channel=1120, mlp=[1024, 512])
        self.fp1 = PointNetFeaturePropagation(in_channel=512, mlp=[512, 512])

        self.conv1 = nn.Conv1d(512, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(512, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(2, 1)
        l0_points = x
        l0_xyz = x[:, :3, :]

        l1_xyz, l1_points = self.sa_1(l0_xyz, l0_points) #[8, 3, 512], [8, 96, 512]
        l2_xyz, l2_points = self.sa_2(l1_xyz, l1_points) #[8, 3, 256], [8, 512, 256]
        l3_xyz, l3_points = self.sa_3(l2_xyz, l2_points) #[8, 3, 128], [8, 512, 128]

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) #[8, 1024, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.relu1(self.bn1(self.conv1(l0_points)))
        x = self.relu2(self.bn2(self.conv2(x)))
        
        return x


'''backbone network for uop'''
class PointNet2(nn.Module):
    def __init__(self):
        super(PointNet2, self).__init__()
        self.input_channels = 3

        self.sa_1 = PointNetSetAbstractionMsg(512, [0.1, 0.2], [16, 32], self.input_channels, [[16, 16, 32], [32, 32, 64], [64, 64, 128]])
        self.sa_2 = PointNetSetAbstractionMsg(256, [0.2, 0.4], [64, 128], 32+64, [[128, 128, 256], [128, 196, 256]])
        self.sa_3 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [128, 256], 256+256, [[128, 128, 256], [128, 196, 256]])
        self.fp3 = PointNetFeaturePropagation(in_channel=256+256+256+256, mlp=[1120, 1024])
        self.fp2 = PointNetFeaturePropagation(in_channel=1120, mlp=[1024, 512])
        self.fp1 = PointNetFeaturePropagation(in_channel=512, mlp=[512, 512])

        self.conv1 = nn.Conv1d(512, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        x = x.transpose(2, 1)
        l0_points = x
        l0_xyz = x[:, :3, :]

        l1_xyz, l1_points = self.sa_1(l0_xyz, l0_points) #[8, 3, 512], [8, 96, 512]
        l2_xyz, l2_points = self.sa_2(l1_xyz, l1_points) #[8, 3, 256], [8, 512, 256]
        l3_xyz, l3_points = self.sa_3(l2_xyz, l2_points) #[8, 3, 128], [8, 512, 128]

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) #[8, 1024, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.relu(self.bn1(self.conv1(l0_points)))
        
        return x


'''PointNet2 modules'''

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points