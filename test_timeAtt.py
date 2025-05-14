"""
Author: Ruojing Li
Date: 2025
"""
import argparse
import os
from data_utils.TestMyDataLoader import TestIRSeqDataLoader
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
from numpy import *
from PIL import Image
from ShootingRules import ShootingRules
from sklearn.metrics import auc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'networks/models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 32]')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch of generator to test [default: None]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--seqlen', type=int, default=100, help='Frame number as an input [default: 100]')
    parser.add_argument('--num_point', type=int, default=100000, help='Point Number [default: 4096]')
    parser.add_argument('--datapath', type=str, default='/autodl-tmp/NUDT-MIRSDT/', help='Data path: /home/ma-user/work/data/NUDT-MIRSDT/')
    parser.add_argument('--log_dir', type=str, default='2023-12-28_21-22__SoftLoUloss_DiffConv1+DGConv234_AttV1_NewTrainDL', help='experiment root')   ## required=True
    parser.add_argument('--visual', action='store_true', default=True, help='visualize result [default: False]')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold of segmentation [default: 0.01]')
    parser.add_argument('--threshold_eval', type=float, default=0.5, help='Threshold in evaluation [default: 0.5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    # args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.epoch is None:
        file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    else:
        file_handler = logging.FileHandler('%s/eval_epoch-%d-g.txt' % (experiment_dir, args.epoch))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.datapath
    NUM_CLASSES = 1
    SEQ_LEN = args.seqlen
    NUM_POINT = args.num_point
    BATCH_SIZE = args.batch_size
    npoint_per = int(args.num_point / args.seqlen)


    print("start loading test data ...")
    TEST_DATASET  = TestIRSeqDataLoader(data_root=root,  seq_len=SEQ_LEN, cat_len=10, transform=None)

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    # detector = torch.nn.DataParallel(MODEL.generator(NUM_CLASSES, SEQ_LEN)).cuda()
    detector = MODEL.generator(NUM_CLASSES, SEQ_LEN).cuda()
    if args.epoch is None:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    else:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/epoch_%d_model.pth' % args.epoch)
    detector.load_state_dict(checkpoint['model_state_dict'])
    detector = detector.eval()
    eval = ShootingRules()

    with torch.no_grad():
        num_batches = 0
        total_intersection_mid = 0
        total_union_mid = 0

        Th_Seg = np.array([0, 1e-3, 1e-2, 1e-1, 0.2, 0.3, .35, 0.4, .45, 0.5, .55, 0.6, .65, 0.7, 0.8, 0.9, 0.95, 1])
        FalseNumAll = np.zeros([len(TEST_DATASET),len(Th_Seg)])
        TrueNumAll = np.zeros([len(TEST_DATASET),len(Th_Seg)])
        TgtNumAll = np.zeros([len(TEST_DATASET),len(Th_Seg)])
        pixelsNumber = np.zeros(len(TEST_DATASET))

        log_string('---- EVALUATION----')

        for seq_idx, seq_dataset in tqdm(enumerate(TEST_DATASET), total=len(TEST_DATASET), smoothing=0.9):
            seq_dataloader = torch.utils.data.DataLoader(seq_dataset, batch_size=BATCH_SIZE, shuffle=False)
            num_batches += len(seq_dataloader)
            seq_midpred_all = []   ## b, t, h, w
            targets_all     = []
            centroids_all   = []
            # if seq_idx < 19:
            #     continue
            for i, (images, targets, centroids, first_end) in enumerate(seq_dataloader):
                images, targets, centroids = images.float().cuda(), targets.float().cuda(), centroids.float().cuda()
                first_frame, end_frame = first_end

                seq_feats, seq_midpred = detector(images)   ## b, t, h, w
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

            ############### for IoU ###############
            pred_choice_mid = (seq_midpred_all.data.cpu().numpy() > args.threshold_eval) * 1.
            batch_label     = targets_all.data.cpu().numpy()
            total_intersection_mid += np.sum(pred_choice_mid * batch_label)
            total_union_mid += ((pred_choice_mid + batch_label) > 0).astype(np.float32).sum()

            ############### for Pd&Fa ###############
            _, t, h, w = targets_all.size()
            pixelsNumber[seq_idx] += t * h * w
            for ti in range(t):
                midpred_ti = seq_midpred_all[:, ti, :, :].data.cpu().numpy().copy()
                centroid_ti  = centroids_all[:, ti, :, :].data.cpu().numpy().copy()
                for th_i in range(len(Th_Seg)):
                    FalseNum, TrueNum, TgtNum = eval(midpred_ti, centroid_ti, Th_Seg[th_i])
                    FalseNumAll[seq_idx, th_i] = FalseNumAll[seq_idx, th_i] + FalseNum
                    TrueNumAll[seq_idx, th_i]  = TrueNumAll[seq_idx, th_i] + TrueNum
                    TgtNumAll[seq_idx, th_i]   = TgtNumAll[seq_idx, th_i] + TgtNum
                # print(ti,TgtNum)

                ############### save results ###############
                if args.visual:
                    midpred_ti = Image.fromarray(uint8(midpred_ti.squeeze(0) * 255))
                    png_name = '%05d.png' % (ti+1)
                    seq_dir = Path(os.path.join(visual_dir, TEST_DATASET.seq_names[seq_idx]))
                    seq_dir.mkdir(exist_ok=True)
                    midpred_ti.save(os.path.join(seq_dir, png_name))


        ############### log Pd&Fa results ###############
        low_idx = [7,13,14,15,16,17,18,19]
        high_idx = [0,1,2,3,4,5,6,8,9,10,11,12]
        Pd_low = np.sum(TrueNumAll[low_idx, :], axis=0) / np.sum(TgtNumAll[:, :], axis=0)
        Fa_low = np.sum(FalseNumAll[low_idx, :], axis=0) / pixelsNumber[low_idx].sum()
        Pd_high = np.sum(TrueNumAll[high_idx, :], axis=0) / np.sum(TgtNumAll[:, :], axis=0)
        Fa_high = np.sum(FalseNumAll[high_idx, :], axis=0) / pixelsNumber[high_idx].sum()
        Pd_all = np.sum(TrueNumAll[:, :], axis=0) / np.sum(TgtNumAll[:, :], axis=0)
        Fa_all = np.sum(FalseNumAll[:, :], axis=0) / pixelsNumber.sum()
        auc_low = auc(Fa_low, Pd_low)
        auc_high = auc(Fa_high, Pd_high)
        auc_all = auc(Fa_all, Pd_all)
        for seq_i in range(len(TEST_DATASET)):
            seq_name = TEST_DATASET.seq_names[seq_i]
            log_string('%s results:\n' % seq_name)
            for seg_i in range(len(Th_Seg)):
                log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[seg_i],
                    TrueNumAll[seq_i, seg_i], TgtNumAll[seq_i, seg_i], TrueNumAll[seq_i, seg_i] / TgtNumAll[seq_i, seg_i],
                    FalseNumAll[seq_i, seg_i], FalseNumAll[seq_i, seg_i] / pixelsNumber[seq_i]))
        log_string('Low SNR results:\tAUC:%.5f\n' % (auc_low))
        for th_i in range(len(Th_Seg)):
            log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i], TrueNumAll[low_idx, th_i].sum(),
                        TgtNumAll[low_idx, th_i].sum(), TrueNumAll[low_idx, th_i].sum() / TgtNumAll[low_idx, th_i].sum(),
                        FalseNumAll[low_idx, th_i].sum(), FalseNumAll[low_idx, th_i].sum() / pixelsNumber[low_idx].sum()))
        log_string('High SNR results:\tAUC:%.5f\n' % (auc_high))
        for th_i in range(len(Th_Seg)):
            log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i], TrueNumAll[high_idx, th_i].sum(),
                        TgtNumAll[high_idx, th_i].sum(), TrueNumAll[high_idx, th_i].sum() / TgtNumAll[high_idx, th_i].sum(),
                        FalseNumAll[high_idx, th_i].sum(), FalseNumAll[high_idx, th_i].sum() / pixelsNumber[high_idx].sum()))
        log_string('Final results:\tAUC:%.5f\n' % (auc_all))
        for th_i in range(len(Th_Seg)):
            log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i], TrueNumAll[:, th_i].sum(),
                        TgtNumAll[:, th_i].sum(), TrueNumAll[:, th_i].sum() / TgtNumAll[:, th_i].sum(),
                        FalseNumAll[:, th_i].sum(), FalseNumAll[:, th_i].sum() / pixelsNumber.sum()))

        ############### log IoU results ###############
        mIoU_mid = total_intersection_mid / total_union_mid
        log_string('Eval avg class IoU of prediction: %f' % (mIoU_mid))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
