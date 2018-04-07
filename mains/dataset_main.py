from __future__ import absolute_import, division, print_function

import sys

import tensorflow as tf

from data_loader.dataset import Dataset
from utils.config import process_config
from utils.utils import get_args


def main():
    args = get_args()
    config = process_config(args.config_file)
    # print(config)
    dataset_train = Dataset(config, 'train')
    dataset_val = Dataset(config, 'validation')
    dataset_test = Dataset(config, 'test')

    with tf.Session() as sess:
        dataset_train.init_iter(sess)
        dataset_val.init_iter(sess)
        for epoch in range(config.trainer.parameters.num_epochs):
            print("\n\n\n New Epoch")
            print("Train:")
            for i in range(dataset_train.num_batch_per_epoch):
                imgs, labels = dataset_train.next_batch(sess)
                print("batch:", i, "shape:", imgs.shape)
            print("Validation:")
            for i in range(dataset_val.num_batch_per_epoch):
                imgs, labels = dataset_val.next_batch(sess)
                print("batch:", i, "shape:", imgs.shape)
        print("test")
        dataset_test.init_iter(sess)
        i = 0
        for i in range(dataset_val.num_batch_per_epoch):
            imgs = dataset_test.next_batch(sess)
            print("batch:", i, "shape:", imgs.shape)


if __name__ == "__main__":
    sys.exit(main())
