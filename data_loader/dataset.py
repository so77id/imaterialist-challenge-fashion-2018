from __future__ import absolute_import, division, print_function

import numpy as np
import multiprocessing

from tqdm import tqdm


import cv2
from base.base_dataset import BaseDataset
from utils.utils import list_to_one_hot, get_filename, get_num_lines

SElF = None

def unwrap_self_load_file(line):
    return Dataset.load_file(SELF, line)


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

        global SELF
        SELF = self
        pool = multiprocessing.Pool(processes=self.config.dataset.parameters.load_processes)

        with tqdm(total=get_num_lines(self.list)) as bar:
            for img_label in pool.imap_unordered(unwrap_self_load_file, file):
                imgs.append(img_label["x"])
                if self.mode != "test":
                    labels_one_hot.append(img_label["y"])
                bar.update(1)

        pool.terminate()
        imgs = np.array(imgs)
        labels_one_hot = np.array(labels_one_hot)

        return {"x": imgs, "y": labels_one_hot}

    def load_file(self, line):
        splited_line = line.split('\n')[0].split(' ')
        # img_id = int(splited_line[0])
        img_path = splited_line[1]
        labels = np.array([int(l) for l in splited_line[2:]])
        video_name = get_filename(img_path)
        if self.config.global_parameters.debug:
            print("Reading img:", video_name)
        img = self.__load_image(img_path)
        if self.mode != "test":
            labels = list_to_one_hot(labels, self.config.dataset.parameters.n_classes)

        return {"x": img, "y": labels}

    def __load_image(self, img_path):
        img = cv2.imread(img_path)
        width_ = self.config.dataset.parameters.width
        height_ = self.config.dataset.parameters.height

        img = cv2.resize(img, (width_, height_))

        return img
