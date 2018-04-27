from models.keras_models.inception_v4 import inception_v4_model

from models.keras_models.resnet_50 import resnet50_model
from models.keras_models.resnet_50_keras import resnet50_keras_model

from models.keras_models.densenet_121 import densenet121_model
from models.keras_models.densenet_121_keras import densenet121_keras_model

from models.keras_models.xception import xception_model

from models.keras_models.inception_resnet_v2 import inception_resnet_v2_model

def model_factory(model_name, img_rows, img_cols, channels, num_classes, dropout_keep_prob=0, checkpoint="", freeze=False):

    if model_name == 'inception_v4':
        model = inception_v4_model(img_rows, img_cols, channels, num_classes, dropout_keep_prob=dropout_keep_prob)
    elif model_name == 'resnet_50':
        # model = resnet50_model(img_rows, img_cols, channels, num_classes)
        model = resnet50_keras_model(img_rows, img_cols, channels, num_classes=num_classes, freeze=freeze, dropout_keep_prob=dropout_keep_prob)
    elif model_name == 'densenet_121':
        # model = densenet121_model(img_rows, img_cols, channels, num_classes=num_classes, dropout_rate=dropout_keep_prob)
        model = densenet121_keras_model(img_rows, img_cols, channels, num_classes=num_classes, freeze=freeze, dropout_keep_prob=dropout_keep_prob)
    elif model_name == 'xception':
        model = xception_model(img_rows, img_cols, channels, num_classes=num_classes, freeze=freeze, dropout_keep_prob=dropout_keep_prob)
    elif model_name == 'inception_resnet_v2':
        model = inception_resnet_v2_model(img_rows, img_cols, channels, num_classes=num_classes, freeze=freeze, dropout_keep_prob=dropout_keep_prob)

    if checkpoint != '':
        print("Loading checkpoint:", checkpoint)
        model.load_weights(checkpoint, by_name=True)

    return model
