import os
import torch
import numpy as np
from PIL import Image
import math
import random
from torch.utils.data import Dataset


class TestSeqDataLoader(Dataset):
    def __init__(self, data_root, samplelist, seq_len=100, transform=None):
        self.data_root = data_root
        self.samplelist = samplelist
        self.seq_len = seq_len
        self.transform = transform
        self.train_mean = 105.4025
        self.train_std = 26.6452

    def __len__(self):
        return len(self.samplelist)

    def get_image_label(self, image_path, label_path):
        image = Image.open(image_path)
        image = np.array(image, dtype=np.float32)
        image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)

        label = Image.open(label_path)
        label = np.array(label, dtype=np.uint8) / 255.
        label = np.expand_dims(label, axis=0)

        centroid = Image.open(label_path.replace('masks', 'masks_centroid'))
        centroid = np.array(centroid, dtype=np.uint8) / 255.
        centroid = np.expand_dims(centroid, axis=0)

        return image, label, centroid

    def sample_sequence(self, idx):
        sample = self.samplelist[idx]   ## frame：各帧在序列中的顺序（0开始）
        for i in range(len(sample)):
            image_path, label_path, frame = sample[i]
            image, label, centroid = self.get_image_label(image_path, label_path)
            if i == 0:
                images = image
                labels = label
                centroids = centroid
            else:
                images = np.concatenate((images, image), axis=1)            ## [c, t, h, w]
                labels = np.concatenate((labels, label), axis=0)            ## [t, h, w]
                centroids = np.concatenate((centroids, centroid), axis=0)   ## [t, h, w]

            if i == 0:
                first_frame = frame
            elif i == len(sample) - 1:
                end_frame = frame

        images = (images - self.train_mean) / self.train_std
        # t, h, w = labels.shape
        # if t < self.seq_len:
        #     images = np.concatenate((images, np.zeros([1, self.seq_len-t, h, w])), axis=1)
        #     labels = np.concatenate((labels, np.zeros([self.seq_len-t, h, w])), axis=0)
        #     centroids = np.concatenate((centroids, np.zeros([self.seq_len-t, h, w])), axis=0)

        # if self.transform is not None:
        #     sample = self.transform(sample)   #########################

        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)
        centroids = torch.from_numpy(centroids)

        return images, labels, centroids, [first_frame, end_frame]

    def __getitem__(self, idx):
        images, labels, centroids, first_end = self.sample_sequence(idx)

        return images, labels, centroids, first_end



class TestIRSeqDataLoader(object):
    def __init__(self, data_root='/gpfs3/LRJ/data/IRSeq', seq_len=100, cat_len=10, transform=None):
        self.data_root = data_root
        self.seq_len = seq_len
        self.cat_len = cat_len
        self.transform = transform
        self.seq_list_file = os.path.join(data_root, 'test.txt')
        self._check_preprocess()
        self.seq_names = list(dict.fromkeys([x.split('/')[0] for x in self.ann_f]))

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]

        image_root = os.path.join(self.data_root, seq_name, 'images')
        label_root = os.path.join(self.data_root, seq_name, 'masks')

        images = np.sort(os.listdir(image_root))
        labels = np.sort(os.listdir(label_root))

        samplelist = []
        num_sample = math.ceil((len(images)-self.cat_len) / (self.seq_len-self.cat_len))
        for i in range(num_sample):
            last_frame = min(len(images), (i+1)*(self.seq_len-self.cat_len)+self.cat_len)
            sample = [(os.path.join(image_root, images[x]), os.path.join(label_root, labels[x]), x)
                      for x in range(max(0, last_frame-self.seq_len), last_frame)]
            samplelist.extend([sample])

        seq_dataset = TestSeqDataLoader(self.data_root, samplelist, self.seq_len, self.transform)

        return seq_dataset

    def _check_preprocess(self):
        if not os.path.isfile(self.seq_list_file):
            print('No such file: {}.'.format(self.seq_list_file))
            return False
        else:
            self.ann_f = np.loadtxt(self.seq_list_file, dtype=bytes).astype(str)
            return True

