"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.TrainDataLoader import TrainIRSeqDataLoader
from data_utils.TestDataLoader import TestIRSeqDataLoader
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'networks/models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def seed_everything(seed=46):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='DeepPro', help='model name [default: pointnet_sem_seg]')  # _MultiScale_v2
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=40, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='3', help='GPU to use [default: GPU 0]')
    parser.add_argument('--gpu_num', type=int, default=1, help='GPU to use')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--datapath', type=str, default='/data/lrj/data/NUDT-MIRSDT-Noise/NUDT-MIRSDT-Noise20.0_FJY(g0.27-o2.5)')
    parser.add_argument('--dataset', type=str, default='NUDT-MIRSDT', help='dataset name [default: NUDT-MIRSDT, RGB-T]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--savepath', type=str, default='./log/', help='Save path [default: ./log/]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--seqlen', type=int, default=40, help='Frame number as an input [default: 100]')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch Size for train generator [default: 128, 72]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--sample_rate', type=int, default=0.1, help='Sampling rate for training [default: 0.1(NUDT-MIRSDT), '
                                                                     '0.03(IRDST), 0.05(RGB-T)]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold of segmentation [default: 0.01]')
    parser.add_argument('--threshold_eval', type=float, default=0.3, help='Threshold in evaluation [default: 0.5]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    if args.gpu_num == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path(args.savepath)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        args.log_dir = args.dataset + '__' + timestr + '__SoftLoUloss_' + args.model + '_DataL' + str(args.seqlen)
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.datapath
    NUM_CLASSES = 1
    SEQ_LEN = args.seqlen
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = TrainIRSeqDataLoader(args.dataset, data_root=root, seq_len=SEQ_LEN, sample_rate=args.sample_rate,
                                         patch_size=args.patch_size, transform=None)  # sample_rate=0.1, 0.03, 0.05
    print("start loading test data ...")
    TEST_DATASET  = TestIRSeqDataLoader(args.dataset, data_root=root,  seq_len=SEQ_LEN, cat_len=int(SEQ_LEN*0.1), transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d sequences" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('networks/models/%s.py' % args.model, str(experiment_dir))

    detector = MODEL.generator(NUM_CLASSES, SEQ_LEN, SEQ_LEN)
    if args.gpu_num > 1:
        detector = torch.nn.DataParallel(detector)#, device_ids=list(np.arange(args.gpu_num)))
    detector = detector.cuda()
    # criterion = MODEL.g_loss().cuda()
    # criterion = MODEL.g_HAMloss().cuda()
    criterion = MODEL.g_SoftLoUloss().cuda()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            detector.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(detector.parameters(), lr=args.learning_rate, momentum=0.9)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch'] + 1
        try:
            detector.module.load_state_dict(checkpoint['model_state_dict'])
        except:
            detector.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    LEARNING_RATE_CLIP = 1e-5
    global_epoch = 0
    best_iou = 0
    ## train
    for epoch in range(start_epoch, args.epoch):
        '''Train'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        num_batches = len(trainDataLoader)
        total_intersection_mid = 0
        total_union_mid = 0
        loss_sum = 0
        detector.train()

        for i, (images, targets) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly = True
            images, targets = images.float().cuda(), targets.float().cuda()

            _, seq_midpred = detector(images)

            loss = criterion(seq_midpred, targets)
            loss.backward()
            optimizer.step()

            seq_midpred = torch.sigmoid(seq_midpred)
            midpred_choice = (seq_midpred.cpu().data.numpy() > args.threshold_eval) * 1.
            batch_label    = targets.cpu().data.numpy()
            total_intersection_mid += np.sum(midpred_choice * batch_label)
            total_union_mid += ((midpred_choice + batch_label)>0).astype(np.float32).sum()
            loss_sum += loss
            # break
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy (IoU) of prediction: %f' % (total_intersection_mid / total_union_mid))

        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epoch:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/epoch_' + str(epoch+1) + '_model.pth'
            log_string('Saving at %s' % savepath)
            if args.gpu_num > 1:
                state = {
                    'epoch': epoch,
                    'model_state_dict': detector.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            else:
                state = {
                    'epoch': epoch,
                    'model_state_dict': detector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate'''
        with torch.no_grad():
            num_batches = 0
            total_intersection_mid = 0
            total_union_mid = 0
            loss_g_sum = 0
            detector = detector.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            # for i, (images, targets) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            for seq_idx, seq_dataset in tqdm(enumerate(TEST_DATASET), total=len(TEST_DATASET), smoothing=0.9):
                if seq_idx % 3 > 0:
                    continue
                seq_dataloader = torch.utils.data.DataLoader(seq_dataset, batch_size=1, shuffle=False)
                num_batches += len(seq_dataloader)
                for i, (images, targets, _, first_end) in enumerate(seq_dataloader):
                    images, targets = images.float().cuda(), targets.float().cuda()

                    _, seq_midpred = detector(images)

                    loss_g_sum += criterion(seq_midpred, targets)

                    seq_midpred = torch.sigmoid(seq_midpred)
                    pred_choice_mid = (seq_midpred.cpu().data.numpy() > args.threshold_eval) * 1.
                    batch_label     = targets.cpu().data.numpy()
                    total_intersection_mid += np.sum(pred_choice_mid * batch_label)
                    total_union_mid += ((pred_choice_mid + batch_label) > 0).astype(np.float32).sum()

            mIoU_mid = total_intersection_mid / total_union_mid
            log_string('Eval mean loss: %f' % (loss_g_sum / float(num_batches)))
            log_string('Eval avg class IoU of prediction: %f' % (mIoU_mid))

            if mIoU_mid >= best_iou:
                best_iou = mIoU_mid
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                if args.gpu_num > 1:
                    state = {
                        'epoch': epoch,
                        'class_avg_iou': mIoU_mid,
                        'model_state_dict': detector.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                else:
                    state = {
                        'epoch': epoch,
                        'class_avg_iou': mIoU_mid,
                        'model_state_dict': detector.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU_mid: %f' % best_iou)

        global_epoch += 1


def path_remake(path):
    return path.replace(' ', '\ ').replace('(', '\(').replace(')', '\)').replace('&', '\&')


if __name__ == '__main__':
    args = parse_args()
    main(args)

    os.system('python test.py --gpu %s --seqlen %d --datapath %s --dataset %s --log_dir %s' % (
            args.gpu, args.seqlen, path_remake(args.datapath), path_remake(args.dataset), path_remake(args.log_dir)))

