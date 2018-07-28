from pprint import pprint
import cv2 as cv
import numpy as np
import random
import os
import time


class DataReader:
    def __init__(self, config):
        self.config = config
        self.index = {
            'train': 0,
            'eval': 0
        }
        self.pools = {}
        self.pools['train'] = []
        self.pools['eval'] = []
        for img_path in os.listdir(config['train_folder']):
            if 'jpg' not in img_path:
                continue
            self.pools['train'].append(img_path)
        for img_path in os.listdir(config['eval_folder']):
            if 'jpg' not in img_path:
                continue
            self.pools['eval'].append(img_path)
        random.shuffle(self.pools['train'])
        random.shuffle(self.pools['eval'])

    def get_data(self, mode):
        return self.__read_img_labels(self.pools[mode], mode)

    def next_batch(self, mode):
        start_idx = self.index[mode]
        end_idx = self.index[mode] + self.config['batch_size']
        datas = self.pools[mode][start_idx:end_idx]

        # complement
        if len(datas) < self.config['batch_size']:
            complement = self.config['batch_size'] - len(datas)
            datas += self.pools[mode][:complement]
            self.index[mode] = 0
            random.shuffle(self.pools[mode])

        # read imgs
        imgs, labels = self.__read_img_labels(datas, mode)
        self.index[mode] = end_idx
        return imgs, labels

    def __read_img_labels(self, paths, mode, cvt_gray=False):
        def one_hot(data):
            one_hot = [0 for _ in range(10)]
            one_hot[int(data)] = 1
            return one_hot

        imgs = []
        labels = []
        for path in paths:
            if mode == 'train':
                base_path = self.config['train_folder']
            else:
                base_path = self.config['eval_folder']
            img_path = base_path + path
            img = cv.imread(img_path)
            if cvt_gray:
                img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            resize_ratio = self.config['resize']
            if resize_ratio != 1:
                img = cv.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
            imgs.append(img)
            if '_' in path:
                path = path[:-6]
            label = [
                one_hot(int(path[0])),
                one_hot(int(path[1])),
                one_hot(int(path[2])),
                one_hot(int(path[3])),
            ]
            labels.append(label)
        labels = np.asarray(labels)
        imgs = np.asarray(imgs)
        if cvt_gray:
            imgs = np.reshape(
                imgs, (-1, imgs[0].shape[0], imgs[0].shape[1], 1))
        else:
            imgs = np.reshape(
                imgs, (-1, imgs[0].shape[0], imgs[0].shape[1], 3))
        return imgs, labels
