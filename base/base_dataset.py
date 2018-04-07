from __future__ import absolute_import, division, print_function

import tensorflow as tf


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
        self.data = self.load_dataset()
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
