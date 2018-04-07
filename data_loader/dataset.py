from __future__ import absolute_import, division, print_function

import numpy as np
import pdb


import cv2
from base.base_dataset import BaseDataset
from utils.utils import list_to_one_hot, get_filename


class Dataset(BaseDataset):

    def __init__(self, config, mode='train'):
        super(Dataset, self).__init__(config, mode)
        self.list_path = "{}/{}".format(config.dataset.original.path, config.dataset.lists.folder)
        self.set_mode()
        self.load_data()

    def set_mode(self):
        if self.mode == 'train':
            self.list = "{}/{}".format(self.list_path, self.config.dataset.lists.train)
        elif self.mode == 'validation':
            self.list = "{}/{}".format(self.list_path, self.config.dataset.lists.validation)
        elif self.mode == 'test':
            self.list = "{}/{}".format(self.list_path, self.config.dataset.lists.test)

    def load_dataset(self):
        print("Loading dataset: {}".format(self.list))
        file = open(self.list, 'r')

        imgs = []
        labels_one_hot = []

        for line in file:
            splited_line = line.split('\n')[0].split(' ')
            # img_id = int(splited_line[0])
            img_path = splited_line[1]
            labels = np.array([int(l) for l in splited_line[2:]])
            video_name = get_filename(img_path)
            if self.config.global_parameters.debug:
                print("Reading img:", video_name)
            img = self.__load_image(img_path)
            imgs.append(img)
            if self.mode != "test":
                labels_one_hot.append(list_to_one_hot(labels, self.config.dataset.parameters.n_classes))
            # else:
            #     labels_one_hot.append(np.zeros(self.config.dataset.parameters.n_classes))

        imgs = np.array(imgs)
        labels_one_hot = np.array(labels_one_hot)

        return {"x": imgs, "y": labels_one_hot}

    def __load_image(self, img_path):
        img = cv2.imread(img_path)
        width_ = self.config.dataset.parameters.width
        height_ = self.config.dataset.parameters.height

        img = cv2.resize(img, (width_, height_))

        return img
