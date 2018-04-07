from __future__ import absolute_import, division, print_function

import os

from utils.config import process_config
from utils.utils import get_args
from utils.dirs import create_dirs

def write_file(filename, dataset):
    file = open(filename, "w")
    for id_name, values in dataset.items():
        filename, labels = values
        line = "{} {}".format(id_name, filename)
        for label in labels:
            line = "{} {}".format(line, label)
        file.write("{}\n".format(line))
    file.close()

def create_list(folder_path, list_path,  mode="train"):
    # Create train list
    dict = {}
    for (dirpath, dirnames, filenames) in os.walk(folder_path):
        filenames.sort()
        for filename in filenames:
            if mode != "test":
                _, id_str, _, labels_str = filename.split('.')[0].split('_')
                labels = [int(l) for l in labels_str.split('-')]
                id = int(id_str)
            else:
                id_str = filename.split('.')[0]
                labels = []
                id = int(id_str)

            dict[id] = ["{}/{}".format(folder_path, filename), labels]


    write_file(list_path, dict)


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config_file)
    except:
        print("missing or invalid arguments")
        exit(0)


    dataset_path = config.dataset.original.path
    train_path = "{}/{}".format(dataset_path, config.dataset.original.train_folder)
    validation_path = "{}/{}".format(dataset_path, config.dataset.original.validation_folder)
    test_path = "{}/{}".format(dataset_path, config.dataset.original.test_folder)

    list_path = "{}/{}".format(dataset_path, config.dataset.lists.folder)
    create_dirs([list_path])

    train_list = "{}/{}".format(list_path, config.dataset.lists.train)
    validation_list = "{}/{}".format(list_path, config.dataset.lists.validation)
    test_list = "{}/{}".format(list_path, config.dataset.lists.test)

    create_list(train_path, train_list)
    create_list(validation_path, validation_list)
    create_list(test_path, test_list, mode="test")



if __name__ == '__main__':
    main()
