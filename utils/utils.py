from __future__ import absolute_import, division, print_function

import argparse
import numpy as np

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c',
        '--config_file',
        metavar='C',
        default='None',
        help='The Configuration file',
    )
    args = argparser.parse_args()
    return args


def get_filename(filename_with_path):
    return str(filename_with_path).split('/')[-1]


def list_to_one_hot(labels, n_classes):
    aux = np.zeros((labels.size, n_classes))
    aux[np.arange(labels.size), labels] = 1

    return aux.sum(axis=0)
