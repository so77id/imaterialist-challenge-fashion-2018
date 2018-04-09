import sys

from utils.scores import fbeta
from utils.metadata import get_metadata_paths
from utils.optimizers import optimizer_factory

from models.keras_models.factory import model_factory
from data_loader.dataset import Dataset

from keras.callbacks import TensorBoard

from utils.config import process_config
from utils.utils import get_args


def main():
    args = get_args()
    config = process_config(args.config_file)

    # Load datasets
    dataset_train = Dataset(config, 'train')
    dataset_val = Dataset(config, 'validation')

    # Get variables
    model_name = config.network.parameters.model_name
    img_rows = int(config.dataset.parameters.height)
    img_cols = int(config.dataset.parameters.width)
    channel = int(config.dataset.parameters.channels)
    num_classes = int(config.dataset.parameters.n_classes)
    dropout_keep_prob = float(config.network.parameters.dropout_keep_prob)
    batch_size = int(config.trainer.parameters.batch_size)
    n_epoch = int(config.trainer.parameters.n_epoch)

    # Get paths
    metadata_path, checkpoint_path, logs_path = get_metadata_paths(config, args)

    # Load model
    model = model_factory(model_name, img_rows, img_cols, channel, num_classes, dropout_keep_prob)

    # Loading optimizer
    optimizer = optimizer_factory(config.trainer.parameters.optimizer)

    # Creating trainner
    model.compile(optimizer=optimizer, loss=config.trainer.parameters.loss_function, metrics=[fbeta])

    # Fit model
    tensorboard = TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=False)

    model.fit(dataset_train.data["x"],
              dataset_train.data["y"],
              batch_size=batch_size,
              epochs=n_epoch,
              shuffle=True,
              verbose=1,
              callbacks=[tensorboard],
              validation_data=(dataset_val.data["x"], dataset_val.data["y"]))
    # Save model
    model.save_weights(checkpoint_path)


if __name__ == "__main__":
    sys.exit(main())
