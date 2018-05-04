import tensorflow
from keras.models import Model
from keras.layers import Input, concatenate

from models.keras_models.factory import model_factory

def emsemble_model(models=[], gpus=[], checkpoints=[], img_rows=224, img_cols=224, channels=3, num_classes=1000, freeze=False, dropout_keep_prob=0.2, use_mvc=False):
    # resnet50 = resnet50_keras_model(img_rows=img_rows,
    #                                 img_cols=img_cols,
    #                                 channels=channels,
    #                                 num_classes=num_classes,
    #                                 freeze=freeze,
    #                                 dropout_keep_prob=dropout_keep_prob,
    #                                 use_mvc=use_mvc,
    #                                 in_model=False)

    assert len(models) != len(gpus) != len(checkpoints)

    input_tensor = Input(shape=(img_rows, img_cols, channels))  # this assumes K.image_data_format() == 'channels_last'

    networks = []
    for model_name, checkpoint, gpu in zip(models, checkpoints, gpus):
        # gpu = /gpu:0 or /gpu:1
        with tf.device(gpu):
            network = model_factory(model_name=model_name,
                                    img_rows=img_rows,
                                    img_cols=img_cols,
                                    channels=channels,
                                    num_classes=num_classes,
                                    dropout_keep_prob=dropout_keep_prob,
                                    checkpoint=checkpoint,
                                    freeze=freeze,
                                    use_mvc=use_mvc,
                                    in_model=False,
                                    input_tensor=input_tensor)
            networks.append(network)

    x = concatenate(networks)
    predictions = Dense(units=num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=predictions, name='Emsemble_trainable')

    return model
