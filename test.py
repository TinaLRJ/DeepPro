"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.TestDataLoader import TestIRSeqDataLoader
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
from numpy import *
import time
from PIL import Image
import scipy.io as scio
from ShootingRules import ShootingRules
from sklearn.metrics import auc
from collections import OrderedDict
from thop import profile, clever_format
# from attribution.core import IR_Integrated_gradient, MeanLinearPath, ZeroLinearPath
from write_results import writeNUDTMIRSDT_ROC, writeMIRST_ROC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'networks/models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 32]')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch of generator to test [default: None]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--seqlen', type=int, default=40, help='Frame number as an input [default: 100]')
    parser.add_argument('--num_point', type=int, default=100000, help='Point Number [default: 4096]')
    parser.add_argument('--datapath', type=str, default='/data/lrj/data/NUDT-MIRSDT-Noise/NUDT-MIRSDT-Noise8.0_FJY(g0.15-o1.3)', help='Data path [default: /data/lrj/data/NUDT-MIRSDT-Noise, /data/lrj/data/IRDST-simulation/]')  ## /gpfs3/LRJ/飞行数据/  IR__2024-03-18_15-00__SoftLoUloss_DiffConv1+DGConv234_AttV1_head1_NewTrainDL, IR__2024-03-17_11-03__SoftLoUloss_DiffConv1+DGConv234_AttV1_head2_NewTrainDL, IR__2024-03-17_11-04__SoftLoUloss_DiffConv1+DGConv234_AttV1_head4_NewTrainDL
    parser.add_argument('--dataset', type=str, default='NUDT-MIRSDT-Noise8.0_FJY(g0.15-o1.3)', help='dataset name [default: NUDT-MIRSDT, IRDST-simulation, RGB-T]')
    parser.add_argument('--log_dir', type=str, default='RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40', help='experiment root')   ## required=True  IR__2024-11-27_10-23__SoftLoUloss_GConv1+DGConv23_AttV1_NewTrainDL40
    parser.add_argument('--logpath', type=str, default='./log/', help='Log path: ./log/')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--threshold_eval', type=float, default=0.5, help='Threshold in evaluation [default: 0.5]')
    parser.add_argument('--attribution', action='store_true', default=False, help='This test is attribution analysis or not')
    return parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)*32/8/1024  # K


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = args.logpath + 'sem_seg/' + args.log_dir
    if args.visual:
        visual_dir = experiment_dir + '/visual/'
        visual_dir = Path(visual_dir)
        visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.epoch is None:
        file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    else:
        file_handler = logging.FileHandler('%s/eval_epoch-%d.txt' % (experiment_dir, args.epoch))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.datapath
    NUM_CLASSES = 1
    SEQ_LEN = args.seqlen
    BATCH_SIZE = args.batch_size


    print("start loading test data ...")
    TEST_DATASET  = TestIRSeqDataLoader(args.dataset, data_root=root,  seq_len=SEQ_LEN, cat_len=int(SEQ_LEN*0.1), transform=None)

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    sys.path.append(experiment_dir)
    MODEL = importlib.import_module(model_name)
    # detector = torch.nn.DataParallel(MODEL.generator(NUM_CLASSES, SEQ_LEN)).cuda()
    detector = MODEL.detector(NUM_CLASSES, SEQ_LEN, SEQ_LEN).cuda()
    if args.epoch is None:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    else:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/epoch_%d_model.pth' % args.epoch)
    # ## multi-GPU models load on single-GPU device
    # new_state_dict = OrderedDict()
    # for k,v in checkpoint['model_state_dict'].items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # detector.load_state_dict(new_state_dict)   ## or use the above detector definition
    # ## ##########################################
    detector.load_state_dict(checkpoint['model_state_dict'])
    detector.eval()
    eval = ShootingRules()

    with torch.no_grad():
        num_batches = 0
        total_intersection_mid = 0
        total_union_mid = 0

        Th_Seg = np.array([0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, .35, 0.4,
                           .45, 0.5, .55, 0.6, .65, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1])
        FalseNumAll = np.zeros([len(TEST_DATASET),len(Th_Seg)])
        TrueNumAll = np.zeros([len(TEST_DATASET),len(Th_Seg)])
        TgtNumAll = np.zeros([len(TEST_DATASET),len(Th_Seg)])
        pixelsNumber = np.zeros(len(TEST_DATASET))


        # if args.attribution:
        #     path_interpolation_func = ZeroLinearPath(fold=50)

        log_string('---- EVALUATION----')

        time_start = time.time()
        for seq_idx, seq_dataset in tqdm(enumerate(TEST_DATASET), total=len(TEST_DATASET), smoothing=0.9):
            seq_dataloader = torch.utils.data.DataLoader(seq_dataset, batch_size=BATCH_SIZE, shuffle=False)
            num_batches += len(seq_dataloader)
            seq_midpred_all = []   ## b, t, h, w
            targets_all     = []
            for i, (images, targets, centroids, first_end) in enumerate(seq_dataloader):
                images, targets = images.float().cuda(), targets.float().cuda()
                first_frame, end_frame = first_end

                if args.attribution:
                    paths = [os.path.join(TEST_DATASET.seq_names[seq_idx], '%05d.png' % (fi+1))
                             for fi in range(first_frame, end_frame+1)]
                    savepath = os.path.join(experiment_dir, 'Attribution_ZeroLinearPath_0.1')
                    # seq_midpred = IR_Integrated_gradient(images, targets, (paths, args.dataset, savepath), detector, path_interpolation_func)

                else:
                    _, seq_midpred = detector(images)   ## b, t, h, w
                    seq_midpred = torch.sigmoid(seq_midpred)

                    if i == 0:
                        seq_midpred_all = seq_midpred
                        targets_all     = targets
                        centroids_all   = centroids
                    else:
                        seq_midpred_all[:, first_frame:last_end+1, :, :] = torch.maximum(seq_midpred_all[:, first_frame:, :, :],
                                                                                         seq_midpred[:, :last_end-first_frame+1, :, :])
                        seq_midpred_all = torch.cat([seq_midpred_all, seq_midpred[:, last_end-first_frame+1:, :, :]], dim=1)
                        targets_all     = torch.cat([targets_all, targets[:, last_end-first_frame+1:, :, :]], dim=1)
                        centroids_all   = torch.cat([centroids_all, centroids[:, last_end-first_frame+1:, :, :]], dim=1)

                    last_first = first_frame
                    last_end = end_frame

            if not args.attribution:
                ############### for IoU ###############
                pred_choice_mid = (seq_midpred_all.data.cpu().numpy() > args.threshold_eval) * 1.
                batch_label     = targets_all.data.cpu().numpy()
                total_intersection_mid += np.sum(pred_choice_mid * batch_label)
                total_union_mid += ((pred_choice_mid + batch_label) > 0).astype(np.float32).sum()

                ############### for Pd&Fa ###############
                _, t, h, w = seq_midpred_all.size()
                pixelsNumber[seq_idx] += t * h * w
                for ti in range(t):
                    midpred_ti = seq_midpred_all[:, ti, :, :].data.cpu().numpy().copy()
                    centroid_ti  = centroids_all[:, ti, :, :].data.cpu().numpy().copy()
                    for th_i in range(len(Th_Seg)):
                        FalseNum, TrueNum, TgtNum = eval(midpred_ti, centroid_ti, Th_Seg[th_i])
                        FalseNumAll[seq_idx, th_i] = FalseNumAll[seq_idx, th_i] + FalseNum
                        TrueNumAll[seq_idx, th_i]  = TrueNumAll[seq_idx, th_i] + TrueNum
                        TgtNumAll[seq_idx, th_i]   = TgtNumAll[seq_idx, th_i] + TgtNum

                    ############### save results ###############
                    if args.visual:
                        midpred_ti_png = Image.fromarray(uint8(midpred_ti.squeeze(0) * 255))
                        plus1 = 0 if args.dataset == 'RGB-T' else 1
                        png_name = '%05d.png' % (ti+1*plus1)
                        seq_dir = Path(os.path.join(visual_dir, TEST_DATASET.seq_names[seq_idx]))
                        seq_dir.mkdir(exist_ok=True)
                        midpred_ti_png.save(os.path.join(seq_dir, png_name))
                        # scio.savemat(os.path.join(seq_dir, '%05d.mat' % (ti+1*plus1)), {'TestOut': midpred_ti.squeeze(0)})

        time_end = time.time()
        # print('FPS=%.3f' % (2000*1.2 / (time_end - time_start)))
        ############### log Pd&Fa results ###############
        if not args.attribution:
            if 'NUDT-MIRSDT' in args.dataset:
                writeNUDTMIRSDT_ROC(FalseNumAll, TrueNumAll, TgtNumAll, pixelsNumber, total_intersection_mid,
                                    total_union_mid, Th_Seg, TEST_DATASET, log_string)
            else:
                writeMIRST_ROC(FalseNumAll, TrueNumAll, TgtNumAll, pixelsNumber, total_intersection_mid,
                               total_union_mid, Th_Seg, TEST_DATASET, log_string)

        flops, params = profile(detector, inputs=(torch.randn(1, 1, args.seqlen, 200, 300).cuda(),))
        flops, params = clever_format([flops, params], '%.3f')
        print('FLOPS for %d frames: ' % SEQ_LEN, flops)
        print('Params:', count_parameters(detector))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
