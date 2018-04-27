from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input

def densenet121_keras_model(img_rows=224, img_cols=224, channels=3, num_classes=1000, freeze=False, dropout_keep_prob=0.2):
    # this could also be the output a different Keras model or layer
    input_tensor = Input(shape=(img_rows, img_cols, channels))  # this assumes K.image_data_format() == 'channels_last'
    # create the base pre-trained model
    base_model = DenseNet121(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    predictions = Dense(units=num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions, name='DenseNet121')

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    if freeze:
        for layer in base_model.layers:
            layer.trainable = False


    return model
