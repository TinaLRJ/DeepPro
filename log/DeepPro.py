import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.basic import TD_Resblock, STD_Resblock
from networks.layers.basic import TDifferenceConv, SDifferenceConv
from networks.layers.TPro import TPro
from networks.losses.HAM_loss_MultiFrame import HAM_loss
from networks.losses.HPM_loss_MultiFrame import HPM_loss



class detector(nn.Module):
    def __init__(self, num_classes, seqlen=100, out_len=100):
        super(detector, self).__init__()
        self.out_len = out_len
        # self.pool1 = nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv_in = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(5,1,1), stride=(1,1,1), padding=(2,0,0)),
                                     nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(TD_Resblock(8, 16), TD_Resblock(16, 32))
        self.layer2 = nn.Sequential(self.pool, TD_Resblock(8, 16), TD_Resblock(16, 32))
        self.layer3 = nn.Sequential(self.pool, TD_Resblock(8, 16), self.pool, TD_Resblock(16, 32))

        self.TPro1 = TPro(d_model=32, num_head=4, seqlen=seqlen, out_len=out_len)
        self.conv_out1_1 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                                       nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.TPro2 = TPro(d_model=32, num_head=4, seqlen=seqlen, out_len=out_len)
        self.conv_out1_2 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                                       nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.TPro3 = TPro(d_model=32, num_head=4, seqlen=seqlen, out_len=out_len)
        self.conv_out1_3 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                                       nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.conv_out2 = nn.Sequential(nn.Conv3d(in_channels=8*3, out_channels=8, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                                       nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.final = nn.Conv3d(in_channels=8, out_channels=num_classes, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))


    def forward(self, seq_imgs):  ## 1.415G
        _,_,_,h,w = seq_imgs.size()
        seq_feats = self.conv_in(seq_imgs)  ## 3.171G     [:, :, :29, :, :]
        seq_feats1 = self.layer1(seq_feats)  ## 20.771G
        seq_feats2 = self.layer2(seq_feats)  ## 20.771G
        seq_feats3 = self.layer3(seq_feats)  ## 20.771G

        seq_feats1 = seq_feats1.permute(0, 3, 4, 1, 2)
        seq_feats1 = self.TPro1(seq_feats1)
        seq_feats1 = self.conv_out1_1(seq_feats1)
        seq_feats2 = seq_feats2.permute(0, 3, 4, 1, 2)
        seq_feats2 = self.TPro2(seq_feats2)
        seq_feats2 = self.conv_out1_2(seq_feats2)
        seq_feats3 = seq_feats3.permute(0, 3, 4, 1, 2)
        seq_feats3 = self.TPro3(seq_feats3)
        seq_feats3 = self.conv_out1_3(seq_feats3)

        b,c,t,h1,w1 = seq_feats1.size()
        _,_,_,h2,w2 = seq_feats2.size()
        _,_,_,h3,w3 = seq_feats3.size()
        seq_feats1 = F.interpolate(seq_feats1.reshape(b,c*t, h1,w1), size=(h,w), mode="bilinear", align_corners=True).reshape(b,c, t, h,w)
        seq_feats2 = F.interpolate(seq_feats2.reshape(b,c*t, h2,w2), size=(h,w), mode="bilinear", align_corners=True).reshape(b,c, t, h,w)
        seq_feats3 = F.interpolate(seq_feats3.reshape(b,c*t, h3,w3), size=(h,w), mode="bilinear", align_corners=True).reshape(b,c, t, h,w)
        seq_feats = self.conv_out2(torch.cat([seq_feats1, seq_feats2, seq_feats3], dim=1))
        seq_midout = self.final(seq_feats)
        seq_midseg = seq_midout.squeeze(dim=1)    ## b, t, h, w

        return seq_feats, seq_midseg



class HAMloss(nn.Module):
    def __init__(self, alpha=[0.1667, 0.8333], gamma=2, MaxClutterNum=39, ProtectedArea=2):
        super(HAMloss, self).__init__()
        self.HAM = HAM_loss(alpha=alpha, gamma=gamma, MaxClutterNum=MaxClutterNum, ProtectedArea=ProtectedArea)

    def forward(self, midpred, target):

        b, t, h, w = midpred.size()
        # input = midpred.view(b*t, h, w).unsqueeze(dim=1)
        # target = target.view(b*t, h, w).unsqueeze(dim=1)
        loss_mid = self.HAM(midpred, target)

        return loss_mid



class HPMloss(nn.Module):
    def __init__(self, alpha=[0.1667, 0.8333], gamma=2, MaxClutterNum=39, ProtectedArea=2):
        super(HPMloss, self).__init__()
        self.HPM = HPM_loss(alpha=alpha, gamma=gamma, MaxClutterNum=MaxClutterNum, ProtectedArea=ProtectedArea)

    def forward(self, midpred, target):

        b, t, h, w = midpred.size()
        # input = midpred.view(b*t, h, w).unsqueeze(dim=1)
        # target = target.view(b*t, h, w).unsqueeze(dim=1)
        loss_mid = self.HPM(midpred, target)

        return loss_mid



class bceloss(nn.Module):
    def __init__(self):
        super(bceloss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, midpred, target):
        loss_mid = self.bce(midpred, target)

        return loss_mid



class SoftLoUloss(nn.Module):
    def __init__(self):
        super(SoftLoUloss, self).__init__()

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



# if __name__ == '__main__':
#     import  torch
#     model = generator(1)
#     seq_imgs = torch.rand(1, 100, 512, 512)
#     (model(seq_imgs))
