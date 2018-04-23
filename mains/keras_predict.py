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

    # Load datasets
    dataset_test = Dataset(config, 'test')

    # Get variables
    model_name = config.network.parameters.model_name
    img_rows = int(config.dataset.parameters.height)
    img_cols = int(config.dataset.parameters.width)
    channel = int(config.dataset.parameters.channels)
    num_classes = int(config.dataset.parameters.n_classes)
    dropout_keep_prob = 1.0  #float(config.network.parameters.dropout_keep_prob)
    label_treshold = float(config.network.predict.parameters.threshold)


    create_dirs([config.network.predict.parameters.predict_path])
    predict_file = "{}/{}".format(config.network.predict.parameters.predict_path, config.network.predict.parameters.predict_file)
    prob_predict_file = "{}/{}".format(config.network.predict.parameters.predict_path, config.network.predict.parameters.prob_predict_file)

    # Load model
    model = model_factory(model_name, img_rows, img_cols, channel, num_classes, dropout_keep_prob, checkpoint=config.network.predict.parameters.checkpoint)

    test_Y = model.predict(dataset_test.data["x"])
    test_Y_ = np.where(test_Y >= label_treshold, 1, 0)

    idxs = np.argsort(dataset_test.data["file_names"])



    # Writing prediction file
    with open(predict_file, 'w') as writer, open(prob_predict_file, 'w') as prob_writer:
        writer.write("id,predicted\n")
        for name, labels, prob_labels in zip(dataset_test.data["file_names"][idxs], test_Y_[idxs], test_Y[idxs]):

            labels_str = " ".join(map(str, np.where(labels == 1)[0]))
            prob_str = " ".join(map(str, prob_labels))

            writer.write("{},{}\n".format(name, labels_str))
            prob_writer.write("{},{}\n".format(name, prob_str))

if __name__ == "__main__":
    sys.exit(main())
