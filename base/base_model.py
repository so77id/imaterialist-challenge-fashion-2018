from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf


class BaseModel:

    def __init__(self, config, models_dir='./model'):
        self.config = config
        self.models_dir = models_dir
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        # loss
        self.loss_map = {}

    # save function thet save the checkpoint in the path defined in configfile
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.models_dir, self.global_step_tensor)

    # print("Model saved")
    # load lateset checkpoint from the experiment path defined in config_file
    def load(self, sess):
        # TODO: Resolve problem with load model
        latest_checkpoint = tf.train.latest_checkpoint(self.models_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just inialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(
                0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(
                self.cur_epoch_tensor, self.cur_epoch_tensor + 1
            )

    # just inialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(
                0, trainable=False, name='global_step'
            )

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    # Optimizer functions
    def train_op(self):
        optimizer = tf.train.GradientDescentOptimizer(
            self.config.trainer.parameters.learning_rate)
        gradients = optimizer.compute_gradients(
            self.loss, tf.trainable_variables())
        train_op = optimizer.apply_gradients(
            gradients, global_step=self.global_step_tensor
        )
        for gradient, variable in gradients:
            tf.summary.histogram("gradients/" + variable.name, gradient)
            tf.summary.histogram("variables/" + variable.name, variable)
        return train_op

    # Losses functions
    def join_losses(self):
        self.loss = np.sum([v for k, v in self.loss_map.items()])

    def append_loss(self, model, name="loss"):
        with tf.name_scope(name) as scope:
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.y, logits=model)
            )
        self.loss_map[name] = loss
