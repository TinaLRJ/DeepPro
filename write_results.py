import numpy as np
from sklearn.metrics import auc


def writeNUDTMIRSDT_ROC(FalseNumAll, TrueNumAll, TgtNumAll, pixelsNumber, total_intersection_mid, total_union_mid,
                        Th_Seg, TEST_DATASET, log_string):
    low_snr = [7,13,14,15,16,17,18,19]
    high_snr = [0,1,2,3,4,5,6,8,9,10,11,12]
    Pd_L = np.sum(TrueNumAll[low_snr, :], axis=0) / np.sum(TgtNumAll[low_snr, :], axis=0)
    Fa_L = np.sum(FalseNumAll[low_snr, :], axis=0) / pixelsNumber[low_snr].sum()
    auc_L = auc(Fa_L, Pd_L)
    Pd_H = np.sum(TrueNumAll[high_snr, :], axis=0) / np.sum(TgtNumAll[high_snr, :], axis=0)
    Fa_H = np.sum(FalseNumAll[high_snr, :], axis=0) / pixelsNumber[high_snr].sum()
    auc_H = auc(Fa_H, Pd_H)


    Pd_all = np.sum(TrueNumAll[:, :], axis=0) / np.sum(TgtNumAll[:, :], axis=0)
    Fa_all = np.sum(FalseNumAll[:, :], axis=0) / pixelsNumber.sum()
    auc_all = auc(Fa_all, Pd_all)
    for seq_i in range(len(TEST_DATASET)):
        seq_name = TEST_DATASET.seq_names[seq_i]
        log_string('%s results:\n' % seq_name)
        for seg_i in range(len(Th_Seg)):
            log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[seg_i],
                TrueNumAll[seq_i, seg_i], TgtNumAll[seq_i, seg_i], TrueNumAll[seq_i, seg_i] / TgtNumAll[seq_i, seg_i],
                FalseNumAll[seq_i, seg_i], FalseNumAll[seq_i, seg_i] / pixelsNumber[seq_i]))


    log_string('Low SNR results:\tAUC:%.5f\n' % (auc_L))
    for th_i in range(len(Th_Seg)):
        log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i],
                                                                      TrueNumAll[low_snr, th_i].sum(),
                                                                      TgtNumAll[low_snr, th_i].sum(),
                                                                      TrueNumAll[low_snr, th_i].sum() / TgtNumAll[
                                                                          low_snr, th_i].sum(),
                                                                      FalseNumAll[low_snr, th_i].sum(),
                                                                      FalseNumAll[low_snr, th_i].sum() / pixelsNumber[
                                                                          low_snr].sum()))
    log_string('High SNR results:\tAUC:%.5f\n' % (auc_H))
    for th_i in range(len(Th_Seg)):
        log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i],
                                                                      TrueNumAll[high_snr, th_i].sum(),
                                                                      TgtNumAll[high_snr, th_i].sum(),
                                                                      TrueNumAll[high_snr, th_i].sum() / TgtNumAll[
                                                                          high_snr, th_i].sum(),
                                                                      FalseNumAll[high_snr, th_i].sum(),
                                                                      FalseNumAll[high_snr, th_i].sum() / pixelsNumber[
                                                                          high_snr].sum()))
    log_string('Final results:\tAUC:%.5f\n' % (auc_all))
    for th_i in range(len(Th_Seg)):
        log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i],
                                                                      TrueNumAll[:, th_i].sum(), TgtNumAll[:, th_i].sum(),
                                                                      TrueNumAll[:, th_i].sum() / TgtNumAll[:, th_i].sum(),
                                                                      FalseNumAll[:, th_i].sum(),
                                                                      FalseNumAll[:, th_i].sum() / pixelsNumber.sum()))

    ############### log IoU results ###############
    mIoU_mid = total_intersection_mid / total_union_mid
    log_string('Eval avg class IoU of prediction: %f' % (mIoU_mid))

    return



def writeMIRST_ROC(FalseNumAll, TrueNumAll, TgtNumAll, pixelsNumber, total_intersection_mid, total_union_mid,
                        Th_Seg, TEST_DATASET, log_string):
    Pd_all = np.sum(TrueNumAll[:, :], axis=0) / np.sum(TgtNumAll[:, :], axis=0)
    Fa_all = np.sum(FalseNumAll[:, :], axis=0) / pixelsNumber.sum()
    auc_all = auc(Fa_all, Pd_all)
    for seq_i in range(len(TEST_DATASET)):
        seq_name = TEST_DATASET.seq_names[seq_i]
        log_string('%s results:\n' % seq_name)
        for seg_i in range(len(Th_Seg)):
            log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[seg_i],
                TrueNumAll[seq_i, seg_i], TgtNumAll[seq_i, seg_i], TrueNumAll[seq_i, seg_i] / TgtNumAll[seq_i, seg_i],
                FalseNumAll[seq_i, seg_i], FalseNumAll[seq_i, seg_i] / pixelsNumber[seq_i]))

    log_string('Final results:\tAUC:%.5f\n' % (auc_all))
    for th_i in range(len(Th_Seg)):
        log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i],
                    TrueNumAll[:, th_i].sum(), TgtNumAll[:, th_i].sum(), TrueNumAll[:, th_i].sum() / TgtNumAll[:, th_i].sum(),
                    FalseNumAll[:, th_i].sum(), FalseNumAll[:, th_i].sum() / pixelsNumber.sum()))

    ############### log IoU results ###############
    mIoU_mid = total_intersection_mid / total_union_mid
    log_string('Eval avg class IoU of prediction: %f' % (mIoU_mid))

    return



