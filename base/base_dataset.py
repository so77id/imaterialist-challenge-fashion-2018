from __future__ import absolute_import, division, print_function

import tensorflow as tf

import os
import h5py
import numpy as np
import time

from utils.dirs import create_dirs


class BaseDataset():

    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode

    def init_iter(self, sess):
        if self.mode != "test":
            feed_dict={self.x: self.data["x"], self.y: self.data["y"]}
        else:
            feed_dict={self.x: self.data["x"]}

        sess.run(
            self.iterator.initializer,
            feed_dict=feed_dict,
        )

    def next_batch(self, sess):
        return sess.run(self.next_element)

    def load_data(self):
        # Loading dataset
        if self.config.dataset.parameters.load_type == "h5":
            start_time = time.time()
            self.data = self.load_h5()
            end_time = time.time()
        else:
            start_time = time.time()
            self.data = self.load_dataset()
            end_time = time.time()

        print("--- %s seconds ---" % (end_time - start_time))

        if self.config.dataset.parameters.load_mode == "keras":
            return

        self.num_batch_per_epoch = (
            self.data["x"].shape[0] // self.config.trainer.parameters.batch_size
        ) + 1
        # Creating TF datasets
        self.x = tf.placeholder(self.data["x"].dtype, self.data["x"].shape)
        if self.mode != "test":
            self.y = tf.placeholder(self.data["y"].dtype, self.data["y"].shape)
            self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        else:
            self.dataset = tf.data.Dataset.from_tensor_slices((self.x))

        self.dataset = self.dataset.prefetch(500)
        self.dataset = self.dataset.shuffle(
            buffer_size=self.config.trainer.parameters.shuffle_buffer_size
        )
        self.dataset = self.dataset.batch(
            self.config.trainer.parameters.batch_size)
        if self.mode != "test":
            self.dataset = self.dataset.repeat(
                self.config.trainer.parameters.num_epochs
            )
        # Creating iterator
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()


    def set_mode(self):
        # Set variables for each especific mode
        raise NotImplementedError

    def load_dataset(self):
        # Return a dataset dictionary {"x": data, "y": one_hot_labels}
        raise NotImplementedError

    def load_h5(self):
        dataset_path = self.config.dataset.original.path
        h5_path = "{}/{}".format(dataset_path, self.config.dataset.folders.h5_folder)

        if self.mode == "train":
            h5_file = "{}/{}-{}.{}".format(h5_path, self.config.dataset.h5.train_pattern, self.config.dataset.parameters.width, "h5")
        elif self.mode == "test":
            h5_file = "{}/{}-{}.{}".format(h5_path, self.config.dataset.h5.test_pattern, self.config.dataset.parameters.width, "h5")
        elif self.mode == "validation":
            h5_file = "{}/{}-{}.{}".format(h5_path, self.config.dataset.h5.validation_pattern, self.config.dataset.parameters.width, "h5")

        if os.path.isfile(h5_file):
            print("Loading {} h5 file: {}".format(self.mode, h5_file))
            h5f = h5py.File(h5_file, 'r')

            x = np.array(h5f['x'])
            y = np.array(h5f['y'], dtype=np.int)

            return {"x": x, "y": y}
        else:
            print("Not exist h5 file: {}".format(self.mode, h5_file))
            self.data = self.load_dataset()
            self.save_h5()
            return self.data


    def save_h5(self):
        dataset_path = self.config.dataset.original.path
        h5_path = "{}/{}".format(dataset_path, self.config.dataset.folders.h5_folder)
        create_dirs([h5_path])

        if self.mode == "train":
            h5_file = "{}/{}-{}.{}".format(h5_path, self.config.dataset.h5.train_pattern, self.config.dataset.parameters.width, "h5")
        elif self.mode == "test":
            h5_file = "{}/{}-{}.{}".format(h5_path, self.config.dataset.h5.test_pattern, self.config.dataset.parameters.width, "h5")
        elif self.mode == "validation":
            h5_file = "{}/{}-{}.{}".format(h5_path, self.config.dataset.h5.validation_pattern, self.config.dataset.parameters.width, "h5")

        print("Creating {} h5 file: {}".format(self.mode, h5_file))
        h5f = h5py.File(h5_file, 'w')
        h5f.create_dataset('x', data=self.data["x"])
        h5f.create_dataset('y', data=self.data["y"])
        h5f.close()
