import tensorflow as tf

from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate


# from models.keras_models.factory import model_factory

def model_factory(model_name, input_tensor, num_classes, dropout_keep_prob, checkpoint, use_imagenet=True):
    if use_imagenet == True:
        weights = 'imagenet'
    else:
        weights = None

    # if model_name == 'inception_v4':
    if model_name == 'resnet_50':
        base_model = ResNet50(input_tensor=input_tensor, weights=weights, include_top=False)
    elif model_name == 'densenet_121':
        base_model = DenseNet121(input_tensor=input_tensor,weights=weights, include_top=False)
    elif model_name == 'densenet_169':
        base_model = DenseNet169(input_tensor=input_tensor,weights=weights, include_top=False)
    elif model_name == 'densenet_201':
        base_model = DenseNet201(input_tensor=input_tensor,weights=weights, include_top=False)
    elif model_name == 'xception':
        base_model = Xception(input_tensor=input_tensor, weights=weights, include_top=False)
    elif model_name == 'inception_resnet_v2':
        base_model = InceptionResNetV2(input_tensor=input_tensor, weights=weights, include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_keep_prob)(x)
    # x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(units=num_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name=model_name)

    return model




def ensemble_model(models=[], gpus=[], checkpoints=[], img_rows=224, img_cols=224, channels=3, num_classes=1000, freeze=False, dropout_keep_prob=0.2, use_mvc=False):
    # assert len(models) == len(gpus) == len(checkpoints)
    #
    #
    # networks = []
    # inputs = []
    # for model_name, checkpoint, gpu in zip(models, checkpoints, gpus):
    #     # gpu = /gpu:0 or /gpu:1
    #     with tf.device(gpu):
    #         print("Loading:", model_name)
    #         input_tensor = Input(shape=(img_rows, img_cols, channels))  # this assumes K.image_data_format() == 'channels_last'
    #         network = model_factory(model_name=model_name,
    #                                 img_rows=img_rows,
    #                                 img_cols=img_cols,
    #                                 channels=channels,
    #                                 num_classes=num_classes,
    #                                 dropout_keep_prob=dropout_keep_prob,
    #                                 checkpoint=checkpoint,
    #                                 freeze=freeze,
    #                                 use_mvc=use_mvc,
    #                                 in_model=False,
    #                                 input_tensor=input_tensor)
    #         networks.append(network)
    #         inputs.append(input_tensor)
    # print("concat")
    #
    # x = concatenate(networks)
    # print("pred")
    # predictions = Dense(units=num_classes, activation='sigmoid')(x)
    #
    # print("model")
    # model = Model(inputs=inputs, outputs=predictions, name='Emsemble_trainable')
    #
    # return model

    assert len(models) == len(gpus) == len(checkpoints)
    print(models, gpus, checkpoints)
    input_tensor = Input(shape=(img_rows, img_cols, channels))  # this assumes K.image_data_format() == 'channels_last'
    networks = []

    for model_name, checkpoint, gpu in zip(models, checkpoints, gpus):
        # gpu = /gpu:0 or /gpu:1
        with tf.device(gpu):
            print("Loading:", model_name)
            network = model_factory(model_name, input_tensor, num_classes, dropout_keep_prob, checkpoint, use_imagenet=False)

            networks.append(network)

    outputs = [network.outputs[0] for network in networks]

    x = Concatenate()(outputs)
    predictions = Dense(units=num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=predictions, name='ensemble')

    return model
