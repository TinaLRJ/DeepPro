import torch
import torch.nn as nn
import torch.nn.functional as F
# from 2023TNNLS Direction-coded Temporal U-shape Module for Multiframe Infrared Small Target Detection


class HPM_loss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=False, MaxClutterNum=39, ProtectedArea=2):
        super(HPM_loss, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(reduce=False)

        self.HardRatio = 1/4
        self.HardNum = round(MaxClutterNum*self.HardRatio)
        self.EasyNum = MaxClutterNum - self.HardNum

        self.MaxClutterNum = MaxClutterNum
        self.ProtectedArea = ProtectedArea
        self.gamma=gamma
        self.alpha=alpha
        self.size_average = size_average
        if isinstance(alpha, (float, int)): self.alpha=torch.Tensor([alpha(0), alpha(1)])
        # if isinstance(alpha, (float, int)): self.alpha = torch.Tensor(alpha)
        if isinstance(alpha, list): self.alpha=torch.Tensor(alpha)


    def forward(self, input, target):   ## Input: [b,100,512,512]    Target: [b,100,512,512]

        b, t, h, w = input.size()

        ## target surrounding = 2
        template = torch.ones(1, 1, 1, 2*self.ProtectedArea+1, 2*self.ProtectedArea+1).to(input.device)  ## [1,1,1,5,5]
        target_prot = F.conv3d(target.unsqueeze(1).float(), template, stride=1,
                               padding=(0,self.ProtectedArea,self.ProtectedArea))         ## [b,1,100,512,512]
        target_prot = (target_prot > 0).squeeze(1).float()          ## [b,100,512,512]

        with torch.no_grad():
            loss_wise = self.bce_loss(input, target.float())        ## learning based on result of loss computing
            loss_p = loss_wise * (1 - target_prot)
            loss_p_sum = torch.sum(loss_p, dim=1, keepdim=True)
            idx = torch.randperm(130) + 20

            batch_l = loss_p.shape[0]
            Wgt = torch.zeros(batch_l, t, h, w)
            for ls in range(batch_l):
                loss_ls = loss_p_sum[ls, :, :, :].reshape(-1)
                loss_topk, indices = torch.topk(loss_ls, 200)
                indices_rand = indices[idx[0:self.HardNum]]         ## random select HardNum samples in top [20-150]
                idx_easy = torch.randperm(len(loss_ls))[0:self.EasyNum].to(input.device)  ## random select EasyNum samples in all image
                indices_rand = torch.cat((indices_rand, idx_easy), 0)
                indices_rand_row = indices_rand // w
                indices_rand_col = indices_rand % w
                Wgt[ls, :, indices_rand_row, indices_rand_col] = 1


            WgtData_New = Wgt.to(input.device)*(1-target_prot) + target.float()           ## [b,100,512,512]
            # WgtData_New = F.conv3d(WgtData_New.unsqueeze(1), template, stride=1,
            #                        padding=(0,self.ProtectedArea,self.ProtectedArea)).squeeze(1)
            WgtData_New[WgtData_New > 0] = 1

        logpt = F.logsigmoid(input)
        logpt_bk = F.logsigmoid(-input)
        pt = logpt.data.exp()
        pt_bk = 1 - logpt_bk.data.exp()
        loss = -self.alpha[1]*(1-pt)**self.gamma*target*logpt - self.alpha[0]*pt_bk**self.gamma*(1-target)*logpt_bk

        loss = loss * WgtData_New

        return loss.sum()



