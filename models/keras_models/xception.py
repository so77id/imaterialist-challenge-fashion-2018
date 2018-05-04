from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input


def xception_model(img_rows=299, img_cols=299, channels=3, num_classes=1000, freeze=False, dropout_keep_prob=0.2, use_mvc=False, input_tensor=None, in_model=True):
    # this could also be the output a different Keras model or layer
    input_tensor = Input(shape=(img_rows, img_cols, channels))  # this assumes K.image_data_format() == 'channels_last'
    # create the base pre-trained model
    base_model = Xception(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_keep_prob)(x)
    # x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)

    if use_mvc:
            predictions = Dense(units=264, activation='sigmoid')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
            print("Loading:", "./mvc_checkpoints/xception_mvc.hdf5")
            model.load_weights("./mvc_checkpoints/xception_mvc.hdf5")

    predictions = Dense(units=num_classes, activation='sigmoid')(x)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    if freeze:
        for layer in base_model.layers:
            layer.trainable = False

    if in_model:
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions, name='Xception')
    else:
        model = predictions

    return model
