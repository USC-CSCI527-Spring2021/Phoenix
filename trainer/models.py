import os
from os import path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, \
    Flatten, Dense, LeakyReLU, add
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Model

from trainer.utils import CHECKPOINT_DIR, create_or_join, RANDOM_SEED

tf.keras.backend.set_image_data_format("channels_last")
np.random.seed(RANDOM_SEED)

def scheduler(epoch, lr):
    '''
    Learning rate scheduler.
    Decrease learning rate base on number of epoch
    :param epoch: current epoch
    :param lr: current learning rate
    :return: new learning rate
    '''
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def residual_block(y, filter, _strides=(1, 1), _project_shortcut=False):
    '''
    Residual Block.
    Implementation of residual block in ResNet
    :param y: input
    :param filter: filter size
    :param _strides: strides
    :param _project_shortcut: identity shortcuts
    :return:
    '''
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = Conv2D(filter, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2D(filter, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Conv2D(filter, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    y = add([shortcut, y])
    y = LeakyReLU()(y)

    return y


def make_or_restore_model(input_shape, model_type, strategy):
    """
    create or restore the model trained before
    :param model: keras model class
    :return: keras model class
    """
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    is_cloud = os.environ.get("TF_KERAS_RUNNING_REMOTELY")

    checkpoint = create_or_join('{}/{}'.format(CHECKPOINT_DIR, model_type))

    if bool(is_cloud):
        if not tf.io.gfile.exists(checkpoint):
            tf.io.gfile.makedirs(checkpoint)
        checkpoints = [path.join(checkpoint, name) for name in tf.io.gfile.listdir(checkpoint)]
    else:
        checkpoints = [path.join(checkpoint, name) for name in os.listdir(checkpoint)]

    # init_model = discard_model(input_shape) if model_type == 'discard' else rcpk_model(input_shape)

    if checkpoints:
        latest_checkpoint = checkpoint
        # latest_checkpoint = max(checkpoints,
        #                         key=lambda x: os.path.getctime(x) if not is_cloud else tf.io.gfile.stat(x).mtime_nsec)
        print("Restoring {} from".format(model_type), latest_checkpoint)
        if strategy == "local":
            model = keras.models.load_model(latest_checkpoint)
        else:
            with strategy.scope():
                # model = discard_model(input_shape) if model_type == 'discarded' else rcpk_model(input_shape)
                model = keras.models.load_model(latest_checkpoint)
            print("Start {} model in distribute mode".format(model_type))
        return model
    print("Creating a new {} model".format(model_type))
    return discard_model(input_shape) if model_type == 'discard' else rcpk_model(input_shape)


def hypertune(hp):
    input_shape = keras.Input((16, 34, 1))
    x = input_shape
    x = Normalization()(x)

    for i in range(hp.Int('num_conv_layer', 1, 5, default=3)):
        x = Conv2D(hp.Int('filters_' + str(i), 32, 512, step=32, default=256),
                   (3, 1),
                   padding="same", data_format="channels_last")(x)
    for i in range(hp.Choice('num_res_block', [5, 10, 20, 30, 40, 50], default=5)):
        x = residual_block(x, hp.Choice('filters_res_block' + str(i), [64, 128, 256, 512], default=256),
                           _project_shortcut=True)
        x = residual_block(x, hp.Choice('filters_res_block' + str(i), [64, 128, 256, 512], default=256),
                           _project_shortcut=True)
    x = Conv2D(kernel_size=1, strides=1, filters=1, padding="same")(x)
    x = Flatten()(x)
    outputs = Dense(34, activation="softmax")(x)
    model = Model(input_shape, outputs)
    model.summary()
    model.compile(
        hp.Choice('optimizer', ['adam', 'sgd', 'Nadam']),
        keras.losses.CategoricalCrossentropy(),
        metrics=keras.metrics.CategoricalAccuracy())
    return model


def discard_model(input_shape):
    """
    Discard Model
    Network structure using idea of CNN
    :param input_shape: data shape
    :return: keras model class
    """
    k_input = keras.Input(input_shape)
    # x = Normalization()(k_input)
    #
    # for _ in range(3):
    #     x = Conv2D(256, (3, 3), padding="same", data_format="channels_last")(x)
    # for _ in range(5):
    #     x = residual_block(x, 256, _project_shortcut=True)
    #     x = residual_block(x, 256, _project_shortcut=True)
    #
    # x = Conv2D(kernel_size=1, strides=1, filters=1, padding="same")(x)
    #
    # x = Flatten()(x)
    # outputs = Dense(34, activation="softmax")(x)
    # model = Model(k_input, outputs)
    model = keras.Sequential()
    model.add(keras.applications.DenseNet201(weights=None, input_tensor=k_input, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(Dense(256))
    model.add(Dense(34, activation="softmax"))
    model.summary()
    model.compile(
        keras.optimizers.Adam(learning_rate=1e-4),
        keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.AUC(multi_label=True),
        ]
    )
    return model


def rcpk_model(input_shape):
    """
    Riichi, Chi, Pon, Kan models
    Network structure using idea of CNN
    :param input_shape: data shape
    :return: keras model class
    """
    k_input = keras.Input(input_shape)
    model = keras.Sequential()
    model.add(keras.applications.DenseNet201(weights=None, input_tensor=k_input, pooling="avg"))
    # model.add(Conv2D(256, (3, 3), padding="same", data_format="channels_last"))
    # model.add(Conv2D(32, (3, 3), padding="same", data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(Dense(256))
    model.add(Dense(2, activation="softmax"))
    # x = Normalization()(k_input)
    # for _ in range(3):
    #     x = Conv2D(256, (3, 3), padding="same", data_format="channels_last")(x)
    # for _ in range(5):
    #     x = residual_block(x, 256, _project_shortcut=True)
    #     x = residual_block(x, 256, _project_shortcut=True)
    # # for _ in range(3):
    # # x = Conv2D(32, (3, 3), padding="same", data_format="channels_last")(x)
    # # x = Conv2D(kernel_size=1, strides=1, filters=1, padding="same")(x)
    # x = Flatten()(x)
    # x = Dense(1024)(x)
    # x = Dense(256)(x)
    # outputs = Dense(2, activation="softmax")(x)
    # model = Model(k_input, outputs)
    model.summary()
    model.compile(
        keras.optimizers.Adam(learning_rate=1e-4),
        keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ])
    return model

