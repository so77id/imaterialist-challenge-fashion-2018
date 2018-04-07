from __future__ import absolute_import, division, print_function

import tensorflow as tf


class BaseTrain:

    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()
        )
        self.sess.run(self.init)

    def train(self):
        for cur_epoch in range(
            self.model.cur_epoch_tensor.eval(self.sess),
            self.config.trainer.parameters.num_epochs + 1,
            1,
        ):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call teh train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
