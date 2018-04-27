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

        # imgs = []
        # labels_one_hot = []
        # file_names = []

        n_lines = get_num_lines(self.list)
        print(n_lines)

        imgs = np.zeros([n_lines, self.config.dataset.parameters.width, self.config.dataset.parameters.height, self.config.dataset.parameters.channels])
        if self.mode != "test":
            labels_one_hot = np.zeros([n_lines, self.config.dataset.parameters.n_classes])
            file_names = np.array([])
        else:
            labels_one_hot = np.array([])
            file_names = np.zeros([n_lines])

        global SELF
        SELF = self
        pool = multiprocessing.Pool(processes=self.config.dataset.parameters.load_processes)

        with tqdm(total=n_lines) as bar:
            for img_label in pool.imap_unordered(unwrap_self_load_file, file):
                # imgs.append(img_label["x"])
                imgs[img_label["img_id"] - 1] = img_label["x"]
                if self.mode != "test":
                    labels_one_hot[img_label["img_id"] - 1] = img_label["y"]
                    # labels_one_hot.append(img_label["y"])
                else:
                    file_names[img_label["img_id"] - 1] = img_label["file_name"]
                    # file_names.append(img_label["file_name"])
                bar.update(1)
        pool.terminate()
        # imgs = np.array(imgs)
        # labels_one_hot = np.array(labels_one_hot)
        # file_names = np.array(file_names)

        return {"x": imgs, "y": labels_one_hot, "file_names": file_names}

    def load_file(self, line):
        splited_line = line.split('\n')[0].split(' ')
        img_id = int(splited_line[0])
        img_path = splited_line[1]
        labels = np.array([int(l) for l in splited_line[2:]])
        video_name = get_filename(img_path).split(".")[0]
        if self.config.global_parameters.debug:
            print("Reading img:", video_name)
        img = self.__load_image(img_path)
        if self.mode != "test":
            labels = list_to_one_hot(labels, self.config.dataset.parameters.n_classes)
            video_name = video_name.split("_")[1]

        return {"x": img, "y": labels, "file_name": int(video_name), "img_id": img_id}

    def __load_image(self, img_path):
        img = cv2.imread(img_path)

        if self.config.dataset.parameters.resize_method == "resize_and_fill":
            img = self.resize_and_fill(img)
        elif self.config.dataset.parameters.resize_method == "resize_and_crop":
            img = self.resize_and_crop(img)

        return img

    def resize_and_fill(self, img):
        fwidth = self.config.dataset.parameters.width
        fheight = self.config.dataset.parameters.height


        height, width, channels = img.shape
        blank_image = np.ones((fheight, fwidth, channels), np.uint8)*255

        if width > height:
            width_ = fwidth
            height_ = (height * width_)//width
        else:
            height_ = fheight
            width_ =  (width * height_)//height

        img = cv2.resize(img, (width_, height_))

        h_offset = (fheight - height_)//2
        w_offset = (fwidth - width_)//2

        blank_image[h_offset:h_offset + height_, w_offset:w_offset + width_] = img

        return blank_image


    def resize_and_crop(self, img):
        fwidth = self.config.dataset.parameters.width
        fheight = self.config.dataset.parameters.height

        height, width, channels = img.shape
        if width > height:
            height_ = fheight
            width_ =  (width * height_)//height
        else:
            width_ = fwidth
            height_ = (height * width_)//width

        img = cv2.resize(img, (width_, height_))

        img = img[(height_ - fheight)//2 : (height_ + fheight)//2, (width_ - fwidth)//2 : (width_ + fwidth)//2]

        return img
