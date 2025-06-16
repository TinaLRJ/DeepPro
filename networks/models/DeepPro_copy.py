import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.basic import Res3D_block, ResTime_block, ResTime_block_v2, ResTime_block_DG, \
    ResTime_block_DD, ResTime_block_S1, ResTime_block_S1_v2, ResTime_block_S1_v3
from networks.layers.basic import DifferenceConv, GradientConv, ResSpace_block, ResSpace_block_v2
from networks.layers.TPro import MultiheadTimeAttention_v1, MultiheadTimeAttention_v2
from networks.layers.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation,PointNetSetAbstraction
from networks.losses.HAM_loss_MultiFrame import HAM_loss
from networks.losses.HPM_loss_MultiFrame import HPM_loss
from networks.losses.HSPM_loss import HSPM_loss
from networks.losses.loss_BCETopKLoss import MyWeightBCETopKLoss
import numpy as np
import cv2


def pc_normalize(pc):
    centroid = torch.mean(pc, axis=1)
    pc = pc - torch.unsqueeze(centroid, dim=1)
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=0)))
    pc = pc / m
    return pc


def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = torch.zeros_like(batch_data).float().to(batch_data.device)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = torch.Tensor([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]]).to(batch_data.device)
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = torch.matmul(shape_pc.reshape(-1, 3), rotation_matrix)
    return rotated_data


def get_points(seq_feats, seq_midout, npoint_per, threshold):
    num_point = npoint_per

    b, t, h, w = seq_midout.size()
    seq_feats = seq_feats.view(b, -1, t, h*w)   ## channel=8
    points = torch.zeros([b, 14, t*num_point]).to(seq_feats.device)
    weights = torch.zeros([b, t, h, w]).to(seq_feats.device)
    positions = torch.zeros([b, 3, t*num_point], dtype=torch.long).to(seq_feats.device)
    for bi in range(b):
        for i in range(t):
            seg_i = seq_midout[bi, i, :, :].view(-1)   ## [h*w]
            _, indices = torch.sort(seg_i, descending=True)
            if seg_i[indices[num_point-1]] < threshold:
                indices = torch.nonzero(seg_i > threshold)
                indices_add = torch.randperm(h*w)[:num_point-len(indices)].to(seq_feats.device)
                indices = torch.cat([indices[:,0], indices_add], dim=0)

            x = indices[:num_point] % w
            y = indices[:num_point] // w
            points[bi, 0, i*num_point:(i+1)*num_point] = x
            points[bi, 1, i*num_point:(i+1)*num_point] = y
            points[bi, 2, i*num_point:(i+1)*num_point] = i
            points[bi, 3:-3, i*num_point:(i+1)*num_point] = seq_feats[bi, :, i, indices[:num_point]]
            points[bi, -3, i*num_point:(i+1)*num_point] = x
            points[bi, -2, i*num_point:(i+1)*num_point] = y
            points[bi, -1, i*num_point:(i+1)*num_point] = i / (t-1)

            weights[bi, i, y, x] = 1

        positions[bi, :, :] = points[bi, :3, :]
        # points[bi, 0, :] -= torch.mean(points[bi, 0, :])   #############################
        # points[bi, 1, :] -= torch.mean(points[bi, 1, :])   #############################
        points[bi, 0:3, :] = pc_normalize(points[bi, 0:3, :])
        points[bi, -3, :] /= torch.max(points[bi, -3, :])
        points[bi, -2, :] /= torch.max(points[bi, -2, :])
        # weights = weights.view(b, t, h, w)

    return points, weights, positions


def get_points_partseg(seq_feats, seq_midout, npoint_per, threshold):
    num_point = npoint_per

    b, t, h, w = seq_midout.size()
    seq_feats = seq_feats.view(b, -1, t, h*w)   ## channel=8
    points = torch.zeros([b, 11, t*num_point]).to(seq_feats.device)
    weights = torch.zeros([b, t, h, w]).to(seq_feats.device)
    positions = torch.zeros([b, 3, t*num_point], dtype=torch.long).to(seq_feats.device)
    for bi in range(b):
        for i in range(t):
            seg_i = seq_midout[bi, i, :, :].view(-1)   ## [h*w]
            _, indices = torch.sort(seg_i, descending=True)
            if seg_i[indices[num_point-1]] < threshold:
                indices = torch.nonzero(seg_i > threshold)
                indices_add = torch.randperm(h*w)[:num_point-len(indices)].to(seq_feats.device)
                indices = torch.cat([indices[:,0], indices_add], dim=0)

            x = indices[:num_point] % w
            y = indices[:num_point] // w
            points[bi, 0, i*num_point:(i+1)*num_point] = x
            points[bi, 1, i*num_point:(i+1)*num_point] = y
            points[bi, 2, i*num_point:(i+1)*num_point] = i
            points[bi, 3:-3, i*num_point:(i+1)*num_point] = seq_feats[bi, :, i, indices[:num_point]]

            weights[bi, i, y, x] = 1

        positions[bi, :, :] = points[bi, :3, :]
        points[bi, 0:3, :] = pc_normalize(points[bi, 0:3, :])

    return points, weights, positions


def get_outseg(x, positions, img_shape):
    ## x: [b, num_points, num_classes]
    ## positions: [b, 3, t*num_point]
    b, t, h, w = img_shape
    seq_outseg = torch.zeros(img_shape).to(x.device)
    b_ind = torch.arange(b).to(x.device).view(b, 1).repeat([1, positions.size(2)])
    seq_outseg[b_ind, positions[:,2,:], positions[:,1,:], positions[:,0,:]] = x[:, :, 0]   ## num_classes=1
    return seq_outseg



class generator(nn.Module):
    def __init__(self, num_classes, seqlen=100, out_len=100):
        super(generator, self).__init__()
        self.out_len = out_len
        # self.pool1 = nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv_in = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(5,1,1), stride=(1,1,1), padding=(2,0,0)),
                                     nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(ResTime_block_S1(8, 16), ResTime_block_S1(16, 32))
        self.layer2 = nn.Sequential(self.pool, ResTime_block_S1(8, 16), ResTime_block_S1(16, 32))
        self.layer3 = nn.Sequential(self.pool, ResTime_block_S1(8, 16), self.pool, ResTime_block_S1(16, 32))

        self.TATT1 = MultiheadTimeAttention_v1(d_model=32, num_head=4, seqlen=seqlen, out_len=out_len)
        self.conv_out1_1 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                                       nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.TATT2 = MultiheadTimeAttention_v1(d_model=32, num_head=4, seqlen=seqlen, out_len=out_len)
        self.conv_out1_2 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                                       nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.TATT3 = MultiheadTimeAttention_v1(d_model=32, num_head=4, seqlen=seqlen, out_len=out_len)
        self.conv_out1_3 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                                       nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.conv_out2 = nn.Sequential(nn.Conv3d(in_channels=8*3, out_channels=8, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                                       nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.final = nn.Conv3d(in_channels=8, out_channels=num_classes, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))


    def forward(self, seq_imgs):  ## 1.415G
        ## 视情况将输入拆分成[1-40, 31-70, 61-100]
        _,_,_,h,w = seq_imgs.size()
        seq_feats = self.conv_in(seq_imgs)  ## 3.171G     [:, :, :29, :, :]
        seq_feats1 = self.layer1(seq_feats)  ## 20.771G
        seq_feats2 = self.layer2(seq_feats)  ## 20.771G
        seq_feats3 = self.layer3(seq_feats)  ## 20.771G
        ## self.conv_in1 + self.layer1 时域感受野为11（上下5帧）

        seq_feats1 = seq_feats1.permute(0, 3, 4, 1, 2)
        seq_feats1 = self.TATT1(seq_feats1)
        seq_feats1 = self.conv_out1_1(seq_feats1)
        seq_feats2 = seq_feats2.permute(0, 3, 4, 1, 2)
        seq_feats2 = self.TATT2(seq_feats2)
        seq_feats2 = self.conv_out1_2(seq_feats2)
        seq_feats3 = seq_feats3.permute(0, 3, 4, 1, 2)
        seq_feats3 = self.TATT3(seq_feats3)
        seq_feats3 = self.conv_out1_3(seq_feats3)

        b,c,t,h1,w1 = seq_feats1.size()
        _,_,_,h2,w2 = seq_feats2.size()
        _,_,_,h3,w3 = seq_feats3.size()
        seq_feats1 = F.interpolate(seq_feats1.reshape(b,c*t, h1,w1), size=(h,w), mode="bilinear", align_corners=True).reshape(b,c, t, h,w)
        seq_feats2 = F.interpolate(seq_feats2.reshape(b,c*t, h2,w2), size=(h,w), mode="bilinear", align_corners=True).reshape(b,c, t, h,w)
        seq_feats3 = F.interpolate(seq_feats3.reshape(b,c*t, h3,w3), size=(h,w), mode="bilinear", align_corners=True).reshape(b,c, t, h,w)
        seq_feats = self.conv_out2(torch.cat([seq_feats1, seq_feats2, seq_feats3], dim=1))
        seq_midout = self.final(seq_feats)
        # seq_midseg = torch.sigmoid(seq_midout).squeeze(dim=1)    ## b, t, h, w
        seq_midseg = seq_midout.squeeze(dim=1)    ## b, t, h, w

        return seq_feats, seq_midseg


class discriminator(nn.Module):
    def __init__(self, num_classes):
        super(discriminator, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(2048, [0.02, 0.05, 0.1], [8, 16, 32], 14, [[8, 8, 16], [16, 16, 32], [32, 32, 48]]) ## npoint(group数量), radius(group半径), nsample(每个group内采样点数), in_channel, mlp
        self.sa2 = PointNetSetAbstractionMsg(512, [0.1, 0.2], [16, 32], 16+32+48, [[48, 48, 64], [48, 48, 64]])
        self.sa3 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 64+64, [[64, 64, 96], [64, 64, 96]])
        self.sa4 = PointNetSetAbstractionMsg(32, [0.4, 0.8], [16, 32], 96+96, [[96, 96, 128], [96, 96, 128]])
        self.fp4 = PointNetFeaturePropagation(128+128+96+96, [128, 128])
        self.fp3 = PointNetFeaturePropagation(64+64+128, [128, 128])
        self.fp2 = PointNetFeaturePropagation(16+32+48+128, [128, 64])
        self.fp1 = PointNetFeaturePropagation(64, [64, 64, 64])
        self.conv1 = nn.Conv1d(64, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(32, num_classes, 1)

    def forward(self, points):
        # xyz = torch.Tensor(1,14,10000).to(seq_imgs.device)
        l0_points = points
        l0_xyz = points[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)   ## [b, num_points, num_classes]
        x = torch.sigmoid(x)
        # seq_outseg = get_outseg(torch.sigmoid(x), positions, weights.size())
        # return seq_outseg, seq_midseg, weights
        return x


class discriminator_partseg(nn.Module):
    def __init__(self, num_classes, normal_channel=True):
        super(discriminator_partseg, self).__init__()
        if normal_channel:
            additional_channel = 8
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = torch.sigmoid(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


class g_HAMloss(nn.Module):
    def __init__(self, alpha=[0.1667, 0.8333], gamma=2, MaxClutterNum=39, ProtectedArea=2):
        super(g_HAMloss, self).__init__()
        self.HAM = HAM_loss(alpha=alpha, gamma=gamma, MaxClutterNum=MaxClutterNum, ProtectedArea=ProtectedArea)

    def forward(self, midpred, target):

        b, t, h, w = midpred.size()
        # input = midpred.view(b*t, h, w).unsqueeze(dim=1)
        # target = target.view(b*t, h, w).unsqueeze(dim=1)
        loss_mid = self.HAM(midpred, target)

        return loss_mid


class g_HPMloss(nn.Module):
    def __init__(self, alpha=[0.1667, 0.8333], gamma=2, MaxClutterNum=39, ProtectedArea=2):
        super(g_HPMloss, self).__init__()
        self.HPM = HPM_loss(alpha=alpha, gamma=gamma, MaxClutterNum=MaxClutterNum, ProtectedArea=ProtectedArea)

    def forward(self, midpred, target):

        b, t, h, w = midpred.size()
        # input = midpred.view(b*t, h, w).unsqueeze(dim=1)
        # target = target.view(b*t, h, w).unsqueeze(dim=1)
        loss_mid = self.HPM(midpred, target)

        return loss_mid


class g_HPMloss_Single(nn.Module):
    def __init__(self, alpha=[0.1667, 0.8333], gamma=2, MaxClutterNum=39, ProtectedArea=2):
        super(g_HPMloss_Single, self).__init__()
        self.HPM = MyWeightBCETopKLoss(alpha=alpha, gamma=gamma, MaxClutterNum=MaxClutterNum, ProtectedArea=ProtectedArea)

    def forward(self, midpred, target):

        b, t, h, w = midpred.size()
        input = midpred.view(b*t, h, w).unsqueeze(dim=1)
        target = target.view(b*t, h, w).unsqueeze(dim=1)
        loss_mid = self.HPM(input, target)

        return loss_mid


class g_HSPMloss(nn.Module):
    def __init__(self, alpha=[0.1667, 0.8333], gamma=2, MaxClutterNum=100, ProtectedArea=2):
        super(g_HSPMloss, self).__init__()
        self.HSPM = HSPM_loss(alpha=alpha, gamma=gamma, MaxClutterNum=MaxClutterNum, ProtectedArea=ProtectedArea)


    def forward(self, midpred, target):
        loss_mid = self.HSPM(midpred, target)

        return loss_mid


class g_loss(nn.Module):
    def __init__(self):
        super(g_loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, midpred, target):
        loss_mid = self.bce(midpred, target)

        return loss_mid


class g_SoftLoUloss(nn.Module):
    def __init__(self):
        super(g_SoftLoUloss, self).__init__()

    def forward(self, midpred, target):
        smooth = 0.00
        midpred = torch.sigmoid(midpred)
        intersection = midpred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(midpred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss_mid = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss_mid = 1 - torch.mean(loss_mid)

        return loss_mid


class g_SoftLoUloss_edgeBlur(nn.Module):
    def __init__(self):
        super(g_SoftLoUloss_edgeBlur, self).__init__()

    def forward(self, midpred, target):
        smooth = 0.00
        midpred = torch.sigmoid(midpred)
        intersection = midpred * target
        target_weight = (torch.sum(target,dim=(2,3))>0).unsqueeze(2).unsqueeze(3) # add by wcl_20240910
        intersection_sum = torch.sum(intersection, dim=(1,2,3))

        template = torch.ones(1, 1, 1, 5, 5).to(target.device)
        target_prot = F.conv3d(target.unsqueeze(1).float(), template, stride=1, padding=(0,2,2))  ## [2,1,512,512]
        target_prot = (target_prot > 0).squeeze(1).float() - target.float()

        # pred_sum = torch.sum(midpred * (1 - target_prot), dim=(1,2,3))
        pred_sum = torch.sum(midpred * (1 - target_prot)*target_weight, dim=(1, 2, 3)) # modified by wcl_20240910
        target_sum = torch.sum(target, dim=(1,2,3))
        loss_mid = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth + 1e-6)

        loss_mid = 1 - torch.mean(loss_mid)

        return loss_mid


class g_weightedSoftLoUloss(nn.Module):
    def __init__(self):
        super(g_weightedSoftLoUloss, self).__init__()
        self.temp = 5
        self.twosides = 0.1

    def forward(self, midpred, target):
        smooth = 0.10
        midpred = torch.sigmoid(midpred)
        intersection = midpred * target

        intersection_sum = torch.sum(intersection, dim=(2,3))
        pred_sum = torch.sum(midpred, dim=(2,3))
        target_sum = torch.sum(target, dim=(2,3))
        loss_mid = 1 - (intersection_sum + smooth) / \
                   (pred_sum + target_sum - intersection_sum + smooth)
        # loss_mid = loss_mid * (target_sum > 0)

        b, t, _, _ = midpred.size()
        Tweight = torch.ones([b,t]).to(midpred.device)
        Tweight[:, 0:int(self.twosides*t/2)] = 0.4
        Tweight[:, -int(self.twosides*t/2):] = 0.4
        weighted_loss = torch.sum(F.softmax(loss_mid * Tweight * self.temp, dim=1) * loss_mid, dim=1)

        return torch.mean(weighted_loss)


class d_loss(nn.Module):
    def __init__(self):
        super(d_loss, self).__init__()
    def forward(self, pred, target, weight):
        loss = F.binary_cross_entropy(pred, target, weight=weight)

        return loss

# if __name__ == '__main__':
#     import  torch
#     model = generator(1)
#     seq_imgs = torch.rand(1, 100, 512, 512)
#     (model(seq_imgs))
