import numpy as np
import sys

from models.keras_models.factory import model_factory
from data_loader.dataset import Dataset

from utils.config import process_config
from utils.utils import get_args
from utils.dirs import create_dirs

def main():
    args = get_args()
    config = process_config(args.config_file)

    prob_np_list = []
    for prob_file in config.prob_files:
        prob_list = []
        with open(prob_file, 'r') as read_file:
            for line in read_file:
                id, labels = line.split(",")
                prob_list.append(np.array(labels.split(" "), dtype=np.float))

        prob_np_list.append(np.array(prob_list))

    prob_np_list = np.array(prob_np_list)

    prob_np_list = prob_np_list.sum(axis=0)
    prob_np_list_ = np.where(prob_np_list > config.threshold, 1, 0)



    # Writing prediction file
    with open(config.predict_file, 'w') as writer, open(config.prob_predict_file, 'w') as prob_writer:
        writer.write("image_id,label_id\n")
        for id, (prob_labels, labels) in enumerate(zip(prob_np_list, prob_np_list_)):

            labels_str = " ".join(map(str, np.where(labels == 1)[0]))
            prob_str = " ".join(map(str, prob_labels))

            writer.write("{},{}\n".format(id+1, labels_str))
            prob_writer.write("{},{}\n".format(id+1, prob_str))


if __name__ == "__main__":
    sys.exit(main())
