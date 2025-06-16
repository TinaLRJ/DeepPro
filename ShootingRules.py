import os
import torch
import torch.nn as nn
import numpy as np
import math
from skimage import measure


class ShootingRules(nn.Module):
    def __init__(self):
        super(ShootingRules, self).__init__()

        return
    def forward(self, output, target, DetectTh):
        FalseNum = 0 #False number
        TrueNum = 0 #True number
        TgtNum = 0
        # DetectTh=0.5 #The detecting threshold used in output


        for i_batch in range(output.shape[0]):
            output_one = output[i_batch,:,:].copy()
            target_one = target[i_batch,:,:].copy()
            # mixdata_one = mixdata_np[i_batch, 0, :, :]

            '''
            fig=plt.figure()
            plt.subplot(221); plt.imshow(np.squeeze(mixdata_one), cmap='gray')
            plt.subplot(222); plt.imshow(np.squeeze(target_one), cmap='gray')
            plt.subplot(223); plt.imshow(np.squeeze(output_one), cmap='gray')
            plt.show()
            '''

            output_one[np.where(output_one < DetectTh)] = 0
            output_one[np.where(output_one >= DetectTh)] = 1

            labelimage = measure.label(target_one, connectivity=2)  # 标记8连通区域
            props = measure.regionprops(labelimage, intensity_image=target_one, cache=True)     #测量标记连通区域的属性

            TgtNum += len(props)
            #####################################################################
            # according to label(the lightest pixels)
            LocLen1 = 2  # 1
            LocLen2 = 5  # 4

            Box2_map = np.ones(output_one.shape)
            for i_tgt in range(len(props)):
                True_flag = 0

                pixel_coords = props[i_tgt].coords
                for i_pixel in pixel_coords:
                    Box2_map[i_pixel[0]-LocLen2:i_pixel[0]+LocLen2+1, i_pixel[1]-LocLen2:i_pixel[1]+LocLen2+1] = 0
                    Tgt_area = output_one[i_pixel[0]-LocLen1:i_pixel[0]+LocLen1+1, i_pixel[1]-LocLen1:i_pixel[1]+LocLen1+1]
                    if Tgt_area.sum() >= 1:
                        True_flag = 1
                if True_flag == 1:
                    TrueNum += 1

            False_output_one = output_one*Box2_map
            FalseNum += np.count_nonzero(False_output_one)


        return  FalseNum, TrueNum, TgtNum




