import cv2
import numpy as np

import h5py
import argparse
import sys
import os

import keras
from keras.utils import np_utils

from utils.config import process_config
from utils.utils import get_args

from data_loader.dataset import Dataset


def main():
    args = get_args()
    config = process_config(args.config_file)

    dataset_train = Dataset(config, 'train')
    dataset_train.save_h5()

    dataset_val = Dataset(config, 'validation')
    dataset_val.save_h5()

    dataset_test = Dataset(config, 'test')
    dataset_test.save_h5()


if __name__ == "__main__":
    sys.exit(main())
