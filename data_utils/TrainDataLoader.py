import os
import torch
import numpy as np
from PIL import Image
import math
import random
from skimage import measure
from torch.utils.data import Dataset


class TrainSeqDataLoader(Dataset):
    def __init__(self, dataset, data_root, samplelist, sample_p, seq_len=100, sample_rate=0.1, patch_size=None, transform=None):
        self.data_root = data_root
        self.samplelist = samplelist
        self.sample_p = sample_p
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.patch_size = patch_size
        self.transform = transform
        self.dataset = dataset
        if 'NUDT-MIRSDT' in dataset:
            self.train_mean = 105.4025
            self.train_std = 26.6452
        elif dataset == 'IRDST-simulation':
            self.train_mean = 106.8523
            self.train_std = 56.9243
        elif dataset == 'RGB-T':
            self.train_mean = 85.0799
            self.train_std = 47.4845
        elif dataset == 'SatVideoIRSDT':
            self.train_mean = 111.47
            self.train_std = 22.43

    def __len__(self):
        return int(len(self.samplelist) * self.sample_rate)

    def get_image_label(self, image_path, label_path):
        image = Image.open(image_path)
        if 'NUDT-MIRSDT' in self.dataset:
            image = image.resize([256, 256])
        # elif self.dataset == 'RGB-T':
        #     image = image.resize([480, 480])
        image = np.array(image, dtype=np.float32)
        image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)

        label = Image.open(label_path)
        if 'NUDT-MIRSDT' in self.dataset:
            label = label.resize([256, 256])
        # elif self.dataset == 'RGB-T':
        #     label = label.resize([480, 480])
        label = np.array(label, dtype=np.float32) / 255.
        label[label > 0] = 1.
        label = np.expand_dims(label, axis=0)

        return image, label

    def sample_sequence(self, idx):
        sample = np.random.choice(self.samplelist, p=self.sample_p)
        # sample = random.choice(self.samplelist, weights=self.sample_p)  ## weights不需要合为1
        for i in range(len(sample)):
            image_path, label_path = sample[i]
            image, label = self.get_image_label(image_path, label_path)
            if i == 0:
                images = image
                labels = label
            else:
                images = np.concatenate((images, image), axis=1)   ## [c, t, h, w]
                labels = np.concatenate((labels, label), axis=0)   ## [t, h, w]

        images = (images - self.train_mean) / self.train_std
        t, h, w = labels.shape
        if t < self.seq_len and idx % 2 == 1:
            images = np.concatenate((images, np.zeros([1, self.seq_len-t, h, w])), axis=1)
            labels = np.concatenate((labels, np.zeros([self.seq_len-t, h, w])), axis=0)
        elif t < self.seq_len and idx % 2 == 0:
            images = np.concatenate((np.zeros([1, self.seq_len-t, h, w]), images), axis=1)
            labels = np.concatenate((np.zeros([self.seq_len-t, h, w]), labels), axis=0)

        if self.patch_size is not None:
            if idx % 2 == 1:
                mid_idx = int(t/2)
            else:
                mid_idx = self.seq_len - math.ceil(t / 2)
            mid_lab = labels[mid_idx, :, :]
            labelimage = measure.label(mid_lab, connectivity=2)  # 标记8连通区域
            props = measure.regionprops(labelimage, cache=True)     #测量标记连通区域的属性
            prob = random.uniform(0,1)
            shake_range = int(self.patch_size / 2 / 3)
            if len(props) > 0 and prob < 0.75:
                tar_idx = torch.randint(0, len(props), [1])[0]
                r0 = int(props[tar_idx].centroid[0] + (torch.rand(1)-0.5) * 2 * shake_range - self.patch_size / 2)
                c0 = int(props[tar_idx].centroid[1] + (torch.rand(1)-0.5) * 2 * shake_range - self.patch_size / 2)
                r0 = min(max(r0, 0), h-self.patch_size-1)
                c0 = min(max(c0, 0), w-self.patch_size-1)
            else:
                r0 = torch.randint(0, h - self.patch_size, [1])[0]
                c0 = torch.randint(0, w - self.patch_size, [1])[0]

            images = images[:, :, r0:r0+self.patch_size, c0:c0+self.patch_size]
            labels = labels[:, r0:r0+self.patch_size, c0:c0+self.patch_size]

        # if self.transform is not None:
        #     sample = self.transform(sample)   #########################

        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)

        return images, labels

    def __getitem__(self, idx):
        images, labels = self.sample_sequence(idx)

        return images, labels



class TrainIRSeqDataLoader(TrainSeqDataLoader):
    def __init__(self, dataset='NUDT-MIRSDT', data_root='./datasets/IRSeq', seq_len=100, sample_rate=0.1, patch_size=None, transform=None):
        if 'NUDT-MIRSDT' in dataset or dataset == 'RGB-T' or dataset == 'SatVideoIRSDT':
            self.seq_list_file = os.path.join(data_root, 'train.txt')
        elif dataset == 'IRDST-simulation':
            self.seq_list_file = os.path.join(data_root, 'img_idx/train_IRDST-simulation.txt')
        self._check_preprocess()
        seq_names = list(dict.fromkeys([x.split('/')[0] for x in self.ann_f]))

        samplelist = []
        sample_p = []
        for seq_name in seq_names:
            if 'NUDT-MIRSDT' in dataset:
                image_root = os.path.join(data_root, seq_name, 'images')
                label_root = os.path.join(data_root, seq_name, 'masks').replace('NUDT-MIRSDT-Noise/'+dataset, 'NUDT-MIRSDT')
                images = np.sort(os.listdir(image_root))
                labels = np.sort(os.listdir(label_root))
            elif dataset == 'IRDST-simulation':
                image_root = os.path.join(data_root, 'images', seq_name)
                label_root = os.path.join(data_root, 'masks', seq_name)
                images = os.listdir(image_root)
                labels = os.listdir(label_root)
                images.sort(key=lambda x:int(x.split('.')[0]))
                labels.sort(key=lambda x:int(x.split('.')[0]))
            if dataset == 'RGB-T':
                image_root = os.path.join(data_root, 'train2017', seq_name, '01')
                label_root = os.path.join(data_root, 'segmentations', seq_name)
                images = np.sort(os.listdir(image_root))
                labels = np.sort(os.listdir(label_root))
            if dataset == 'SatVideoIRSDT':
                image_root = os.path.join(data_root, 'train', seq_name, 'img')
                label_root = os.path.join(data_root, 'train', seq_name, 'mask')
                images = np.sort(os.listdir(image_root))
                labels = np.sort(os.listdir(label_root))


            for i in range(int(seq_len*0.1), len(images)):
                sample = [(os.path.join(image_root, images[x]), os.path.join(label_root, labels[x]))
                          for x in range(max(0, i-seq_len), i)]
                samplelist.extend([sample])
                sample_p.append(len(sample))

        sample_p = [p/sum(sample_p) for p in sample_p]
        super(TrainIRSeqDataLoader, self).__init__(dataset, data_root, samplelist, sample_p, seq_len, sample_rate, patch_size, transform)

    def _check_preprocess(self):
        if not os.path.isfile(self.seq_list_file):
            print('No such file: {}.'.format(self.seq_list_file))
            return False
        else:
            self.ann_f = np.loadtxt(self.seq_list_file, dtype=bytes).astype(str)
            return True

