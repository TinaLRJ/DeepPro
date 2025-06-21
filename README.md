# Probing Deep into Temporal Profile Makes the Infrared Small Target Detector Much Better

Pytorch implementation of our deep temporal probe network (DeepPro).&nbsp;[**[Paper]**](https://arxiv.org/pdf/2506.12766)

Our work is currently under peer review. The contributions of this work are as follows:

* We reveal some new insights from a more crucial profile (i.e., temporal profile): **long-term temporal information is much more essential for IRST detection**, which includes global temporal saliency of target signals and correlation information between different signals. We validated the importance of the temporal profile in IRST detection, by developing the first predictive attribution tool.
* Inspired by our research, we remodel the IRST detection task as **a one-dimensional signal anomaly detection task**. Then, we propose an IRST detection network (i.e., DeepPro) to leverage the essential temporal profile information with multiply-add operations only in the time dimension.
* Experimental results show that DeepPro not only achieves **a significant performance improvement in IRST detection with extremely high efficiency and continuity**, but also has high robustness to dim targets and scenes with strong noise.


## Requirements
- Python 3
- torch
- tqbm
<br><br>

## Datasets

NUDT-MIRSDT &nbsp; [[download dir]](https://mail.nudt.edu.cn/coremail/common/nfFile.jsp?share_link=5418DA3CCDBA48F8A81AE154DA02C3D5&uid=liruojing%40nudt.edu.cn) (Extraction code: M6PZ)
is a synthesized dataset, which contains 120 sequences. We use 80 sequences for training and 20 sequences for test.
We divide the test set into two subsets according to their SNR ((0, 3], (3, 10)).

In the test set, targets in 8 sequences are so weak (SNR lower than 3). It is very challenging to detect these targets. The test set includes Sequence[47, 56, 59, 76, 92, 101, 105, 119].

Other datasets include NUDT-MIRSDT-Noise dataset, IRSDT dataset and [RGBT-Tiny](https://github.com/XinyiYing/RGBT-Tiny) dataset.

## SatVideoIRSDT Dataset for SatVideoIRSTD Challenge &nbsp;[**[Homepage]**](https://videoirstd.github.io/)

Training set &nbsp; [[download dir]](https://pan.baidu.com/s/1qZn33wG7xDCjpFImPBTY5Q?pwd=8tvk) includes 1000 sequences. Validation set &nbsp; [[download dir]](https://pan.baidu.com/s/1ab3vUmIrwF9VAZSgc2EF3g?pwd=eqf2) includes 200 sequences.



## Train
```bash
python train.py --model 'DeepPro' --seqlen 40 --dataset [dataset name] --datapath [dataset path]
python train.py --model 'DeepPro-Plus' --seqlen 40 --dataset [dataset name] --datapath [dataset path]
```
<br>


## Test
```bash
python test.py --seqlen 40 --datapath [dataset path] --dataset [dataset name] --logpath [log path] --log_dir [trained model path]
python test.py --seqlen 40 --datapath './datasets/SatVideoIRSDT' --dataset 'SatVideoIRSDT' --logpath './log/' --log_dir 'SatVideoIRSDT__2025-06-19_18-10__SoftLoUloss_DeepPro-Plus_DataL40'  # test for SatVideoIRSTD challenge
```
<br>


## Results and Trained Models

#### Quantative Results 

The comparison results of computational complexity and computational efficiency are as follows,

| Model         | Params | FPS | GFLOPs (480*720) |
| ------------- |:------:|:---:|:----------------:|
| Res-UNet+DTUM | 1165 KB | 25.39 | 54.0 |
| STDMANet | 46404 KB | 5.16 | 503.8 |
| Res-U+RFR | 3980 KB | 34.77 | 48.2 |
| DeepPro | 192 KB | 155.40 | 5.3 |
| DeepPro-Plus | 277 KB | 185.22 | 20.5 |


on NUDT-MIRSDT (SNR≤3)

| Model         | Pd (x10(-2))|  Fa (x10(-5)) | AUC |  |
| ------------- |:-----------:|:-------------:|:---:|:------:|
| Res-UNet+DTUM | 91.68 | 2.37 | 0.9921 | [[Weights]](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM/blob/main/results/ResUNet_DTUM_SpatialDeepSupFalse_fullySup/ResUNet_DTUM.pth) |
| STDMANet | 92.82 | 2.88 | 0.9860 |
| Res-U+RFR | 64.65 | 24.09 | 0.8698 |
| DeepPro | 95.84 | 0.52 | 0.9952 | [[Weights]](https://github.com/TinaLRJ/DeepPro/tree/main/log/sem_seg/NUDT-MIRSDT__2024-12-28_16-21__SoftLoUloss_DeepPro_DataL40/checkpoints/best_model.pth) |
| DeepPro-Plus | 99.24 | 1.65 | 0.9955 | [[Weights]](https://github.com/TinaLRJ/DeepPro/tree/main/log/sem_seg/NUDT-MIRSDT__2024-12-28_16-21__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth) |


on NUDT-MIRSDT (all)

| Model         | Pd (x10(-2))|  Fa (x10(-5)) | AUC ||
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| Res-UNet+DTUM | 97.46 | 3.00 | 0.9967 | [[Weights]](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM/blob/main/results/ResUNet_DTUM_SpatialDeepSupFalse_fullySup/ResUNet_DTUM.pth) |
| STDMANet | 96.59 | 3.40 | 0.9908 |
| Res-U+RFR | 88.61 | 11.58 | 0.9502 |
| DeepPro | 98.50 | 0.72 | 0.9973 | [[Weights]](https://github.com/TinaLRJ/DeepPro/tree/main/log/sem_seg/NUDT-MIRSDT__2024-12-28_16-21__SoftLoUloss_DeepPro_DataL40/checkpoints/best_model.pth) |
| DeepPro-Plus | 99.71 | 2.69 | 0.9978 | [[Weights]](https://github.com/TinaLRJ/DeepPro/tree/main/log/sem_seg/NUDT-MIRSDT__2024-12-28_16-21__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth) |


on NUDT-MIRSDT-Noise

| Model         | Pd (x10(-2))|  Fa (x10(-5)) | AUC ||
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| Res-UNet+DTUM | 43.90 | 4.86 | 0.9413 |
| STDMANet | 51.65 | 1.95 | 0.8766 |
| Res-U+RFR | 35.11 | 464.92 | 0.8655 |
| DeepPro | 59.17 | 1.76 | 0.9638 | [[Weights]](https://github.com/TinaLRJ/DeepPro/tree/main/log/sem_seg/NUDT-MIRSDT-Noise8.0_FJY(g0.15-o1.3)__2024-12-27_23-28__SoftLoUloss_DeepPro_DataL40/checkpoints/best_model.pth) |
| DeepPro-Plus | 76.23 | 1.69 | 0.9171 | [[Weights]](https://github.com/TinaLRJ/DeepPro/tree/main/log/sem_seg/NUDT-MIRSDT-Noise8.0_FJY(g0.15-o1.3)__2024-12-27_23-28__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth) |


## Citiation
```
@misc{li2025probingdeeptemporalprofile,
      title={Probing Deep into Temporal Profile Makes the Infrared Small Target Detector Much Better}, 
      author={Ruojing Li and Wei An and Xinyi Ying and Yingqian Wang and Yimian Dai and Longguang Wang and Miao Li and Yulan Guo and Li Liu},
      year={2025},
      eprint={2506.12766},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.12766}, 
}
```
<br>

## Contact
Welcome to raise issues or email to [liruojing@nudt.edu.cn](liruojing@nudt.edu.cn) for any question.
