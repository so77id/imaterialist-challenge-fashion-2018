from __future__ import absolute_import, division, print_function
import cv2
import keras

import numpy as np
import multiprocessing

from tqdm import tqdm

from utils.utils import list_to_one_hot, get_num_lines
from utils.thread_safe import threadsafe_generator

SElF = None


def unwrap_self_load_file(line):
    return DatasetGenerator.load_file(SELF, line)


class DatasetGenerator(object):
    def __init__(self, config, mode, shuffle=True):
        'Initialization'

        self.config = config
        self.mode = mode
        self.set_mode()

        self.dim = (self.config.dataset.parameters.width, self.config.dataset.parameters.height)
        self.batch_size = self.config.trainer.parameters.batch_size
        self.num_epochs = self.config.trainer.parameters.num_epochs
        self.n_channels = self.config.dataset.parameters.channels
        self.n_classes = self.config.dataset.parameters.n_classes
        self.shuffle = shuffle

        #
        self.paths, self.labels = self.load_dataset_list()
        self.list_IDs = np.arange(len(self.paths))

        self.len = self.__len__()

    def set_mode(self):
        list_path = "{}/{}".format(self.config.dataset.original.path, self.config.dataset.lists.folder)
        if self.mode == 'train':
            self.list = "{}/{}".format(list_path, self.config.dataset.lists.train)
        elif self.mode == 'validation':
            self.list = "{}/{}".format(list_path, self.config.dataset.lists.validation)
        elif self.mode == 'test':
            self.list = "{}/{}".format(list_path, self.config.dataset.lists.test)

    def load_dataset_list(self):
        print("Loading dataset list: {}".format(self.list))
        file = open(self.list, 'r')

        paths = []
        labels_one_hot = []

        global SELF
        SELF = self
        pool = multiprocessing.Pool(processes=self.config.dataset.parameters.load_processes)

        with tqdm(total=get_num_lines(self.list)) as bar:
            for img_path, label in pool.imap_unordered(unwrap_self_load_file, file):
                paths.append(img_path)
                if self.mode != "test":
                    labels_one_hot.append(label)
                bar.update(1)

        labels_one_hot = np.array(labels_one_hot)

        return paths, labels_one_hot

    def load_file(self, line):
        splited_line = line.split('\n')[0].split(' ')
        # img_id = int(splited_line[0])
        img_path = splited_line[1]
        labels = np.array([int(l) for l in splited_line[2:]])
        # video_name = get_filename(img_path)
        # if self.config.global_parameters.debug:
        #     print("Reading img:", video_name)
        # img = self.__load_image(img_path)
        if self.mode != "test":
            labels = list_to_one_hot(labels, self.config.dataset.parameters.n_classes)

        return img_path, labels

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(self.list_IDs.size)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, :] = self.__load_image(self.paths[ID])

            # Store class
            y[i] = self.labels[ID]
        
        return (X, y)

    def __load_image(self, img_path):
        img = cv2.imread(img_path)
        width_ = self.config.dataset.parameters.width
        height_ = self.config.dataset.parameters.height

        img = cv2.resize(img, (width_, height_))

        return img

    @threadsafe_generator
    def generate(self):
        'Generates batches of enzymes from list_enzymes'
        # Infinite loop
        for e in range(int(self.num_epochs)):
            # Generate order of exploration of augmented dataset
            self.__get_exploration_order()

            # Generate batches
            imax = self.__len__()
            for i in range(imax):
                # Find local list of enzymes
                X, y = self.__getitem__(i)

                yield X, y

    def __get_exploration_order(self):
        'Generates indexes of exploration'
        # Find exploration order
        self.indexes = np.arange(self.list_IDs.size)
        if self.shuffle:
            np.random.shuffle(self.indexes)

        return self.indexes
