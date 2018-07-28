import cv2 as cv
import numpy as np
import random
import os
import re


class DataReader:
    def __init__(self, config, mode=None):
        self.config = config
        self.index = {
            'train': 0,
            'eval': 0
        }

        self.exclude_folder = ['.DS_Store']

        self.where_label_is = config['where_label_is']

        self.img_path_pool = {
            'train': [],
            'test': []
        }

        self.train_path = config['train_folder']
        self.test_path = config['test_folder']

        self.finish_all_data = False
        self.data_size = None

        self.number_n_batch = 0

        self.mode = mode
        if mode is None:
            self.__collect_train_filepath()
            self.__collect_test_filepath()

        elif mode == 'train':
            self.__collect_train_filepath()

        elif mode == 'test':
            self.__collect_test_filepath()

    def __collect_train_filepath(self):
        if self.where_label_is == 'file_with_label':
            for filename in os.listdir(self.train_path):
                if 'jpg' not in filename:
                    continue
                file_path = os.path.join(self.train_path, filename)
                self.img_path_pool['train'].append(file_path)

        elif self.where_label_is == 'folder_with_label':
            for foldername in os.listdir(self.train_path):
                if foldername in self.exclude_folder:
                    continue
                folder_path = os.path.join(self.train_path, foldername)
                for filename in os.listdir(folder_path):
                    if 'jpg' not in filename:
                        continue
                    file_path = os.path.join(folder_path, filename)
                    self.img_path_pool['train'].append(file_path)

        random.shuffle(self.img_path_pool['train'])

        self.train_size = len(self.img_path_pool['train'])

    def __collect_test_filepath(self):
        if self.where_label_is == 'file_with_label':
            for filename in os.listdir(self.test_path):
                if 'jpg' not in filename:
                    continue
                file_path = os.path.join(self.test_path, filename)
                self.img_path_pool['test'].append(file_path)

        elif self.where_label_is == 'folder_with_label':
            for foldername in os.listdir(self.test_path):
                if foldername in self.exclude_folder:
                    continue
                folder_path = os.path.join(self.test_path, foldername)
                for filename in os.listdir(folder_path):
                    if 'jpg' not in filename:
                        continue
                    file_path = os.path.join(folder_path, filename)
                    self.img_path_pool['test'].append(file_path)

        random.shuffle(self.img_path_pool['test'])

        self.test_size = len(self.img_path_pool['test'])

    def next_batch(self):
        mode = self.mode
        start_idx = self.index[mode]
        end_idx = self.index[mode] + self.config['batch_size']
        batch_pool = self.img_path_pool[mode][start_idx:end_idx]

        # complement
        if len(batch_pool) < self.config['batch_size']:
            complement = self.config['batch_size'] - len(batch_pool)
            batch_pool += self.img_path_pool[mode][:complement]
            self.index[mode] = self.data_size
            random.shuffle(self.img_path_pool[mode])

        # read imgs
        if self.where_label_is == 'file_with_label':
           imgs, labels = self.__read_filename_with_label(batch_pool)

        elif self.where_label_is == 'folder_with_label':
            imgs, labels = self.__read_folder_with_label(batch_pool)

        self.index[mode] = end_idx

        if self.index[mode] == self.data_size:
            self.finish_all_data = True

        self.number_n_batch += 1

        return imgs, labels

    def __read_folder_with_label(self, path_pool, cvt_gray=False):
        imgs = []
        labels = []
        for img_path in path_pool:
            if self.mode == 'train':
                base_path = self.train_path
            else:
                base_path = self.test_path

            path_trim = img_path.replace(base_path, '')
            split_idx = re.search('/', path_trim).span()[0]
            label = int(path_trim[:split_idx])
            labels.append(label)

            img = cv.imread(img_path)
            if cvt_gray:
                img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

            resize_ratio = self.config['resize']
            if resize_ratio != 1:
                img = cv.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)

            imgs.append(img)

        labels = np.asarray(labels)
        imgs = np.asarray(imgs)

        if cvt_gray:
            imgs = np.reshape(
                imgs, (-1, imgs[0].shape[0], imgs[0].shape[1], 1))
        else:
            imgs = np.reshape(
                imgs, (-1, imgs[0].shape[0], imgs[0].shape[1], 3))

        return imgs, labels

    def __read_filename_with_label(self, path_pool, cvt_gray=False):
        def one_hot(data):
            one_hot = [0 for _ in range(10)]
            one_hot[int(data)] = 1
            return one_hot

        imgs = []
        labels = []
        for img_path in path_pool:
            img = cv.imread(img_path)

            if cvt_gray:
                img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

            resize_ratio = self.config['resize']
            if resize_ratio != 1:
                img = cv.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)

            imgs.append(img)

            filename = img_path.replace(self.train_path, '').replace(self.test_path, '')

            if '_' in filename:
                filename = filename[:-6]
            label = [
                one_hot(int(filename[0])),
                one_hot(int(filename[1])),
                one_hot(int(filename[2])),
                one_hot(int(filename[3])),
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
