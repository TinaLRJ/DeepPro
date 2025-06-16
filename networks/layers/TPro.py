import math
import torch
import torch.nn as nn
from typing import List
from torch.nn import init
import torch.nn.functional as F


class TPro(nn.Module):
    def __init__(self, d_model=32, num_head=8, seqlen=100, out_len=100):
        super(TPro, self).__init__()

        self.seqlen = seqlen
        self.num_head = num_head
        self.hidden_dim = d_model // num_head
        self.out_len = out_len

        QK_heads: List[nn.Module] = []
        for i in range(num_head):
            QK_heads.append(nn.Linear(seqlen, self.out_len))

        self.QK_heads = nn.Sequential(*QK_heads)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm3d(d_model)
        self.conv = nn.Sequential(nn.Conv3d(in_channels=d_model, out_channels=d_model, kernel_size=(1,1,1), stride=(1,1,1)),
                                  nn.BatchNorm3d(d_model), nn.ReLU(inplace=True))


    def forward(self, input):  # size: (1, h, w, 8*4, time_length)
        bs, h, w, d, slen = input.size()
        value = input / (self.seqlen**0.5)
        value = value.view(bs, h, w, self.num_head, self.hidden_dim, slen)

        qkv = torch.zeros([bs, h, w, self.num_head, self.hidden_dim, self.out_len]).to(input.device)
        for i in range(self.num_head):
            qkv[:,:,:,i,:,:] = self.QK_heads[i](value[:,:,:,i,:,:])

        qkv = qkv.view(bs, h, w, -1, self.out_len).permute(0, 3, 4, 1, 2)
        qkv = self.relu(self.norm1(qkv))
        x = self.conv(qkv)
        return x

'''
seq_names = ['Sequence85', 'Sequence86', 'Sequence87', 'Sequence88', 'Sequence89', 'Sequence90', 'Sequence91', 'Sequence92', 'Sequence93', 'Sequence94',
             'Sequence95', 'Sequence96', 'Sequence97', 'Sequence47', 'Sequence56', 'Sequence59', 'Sequence76', 'Sequence101', 'Sequence105', 'Sequence119']
seq_i = 19
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.io as scio
matplotlib.use('agg')
savepath = '/data/lrj/PPA-Net/log/sem_seg/IR__2024-11-27_10-23__SoftLoUloss_GConv1+DGConv23_AttV1_NewTrainDL40/visual_TP/'
seq_path = Path(os.path.join(savepath, seq_names[seq_i]))
os.makedirs(seq_path, exist_ok=True)

seq_feats = torch.norm(value[:,:,:,i,:,:], dim=3, p=2)
seq_feats_o = torch.norm(qkv[:,:,:,i,:,:], dim=3, p=2)
for ti in range(40):
    # channel_0 = seq_feats_0.shape[1]
    # for ci in range(channel_0):
    plt.figure()
    plt.imshow(seq_feats.data.cpu().numpy()[0, :, :, ti], vmin=0,
               vmax=seq_feats[0, :, :, ti].max(), cmap=matplotlib.cm.jet)
    plt.colorbar()
    plt.savefig(os.path.join(seq_path, ('%05d_tp%d_input.png' % (ti + 1, i))))
    plt.close()

    # channel_1 = seq_feats_1.shape[1]
    # for ci in range(channel_1):
    plt.figure()
    plt.imshow(seq_feats_o.data.cpu().numpy()[0, :, :, ti], vmin=0,
               vmax=seq_feats_o[0, :, :, ti].max(), cmap=matplotlib.cm.jet)
    plt.colorbar()
    plt.savefig(os.path.join(seq_path, ('%05d_tp%d_output.png' % (ti + 1, i))))
    plt.close()

savepath = '/data/lrj/PPA-Net/log/sem_seg/NUDT-MIRSDT-Noise8.0_FJY(g0.15-o1.3)__2024-12-28_16-27__SoftLoUloss_1*1Conv_MultiScale_v2_AttV1_Head4_NewTrainDL20/visual_TP/tp_weight'
os.makedirs(savepath, exist_ok=True)
for tpi in range(self.num_head):
    tp = self.QK_heads[tpi].weight.data
    plt.figure()
    plt.imshow(tp.data.cpu().numpy(), cmap=matplotlib.cm.jet)
    plt.colorbar()
    savename = savepath + '/tp%d_3.png' % tpi
    plt.savefig(savename)
    # plt.savefig('/data/lrj/PPA-Net/log/sem_seg/IR__2024-11-27_10-23__SoftLoUloss_GConv1+DGConv23_AttV1_NewTrainDL40/visual_TP/tp_weight/tp%d.png' % tpi)
    plt.close()
    save_name = savename.replace('.png', '.mat')
    scio.savemat(save_name, {'map_attribution': tp})

'''


