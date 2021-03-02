import json
import os
import subprocess
from os import path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, \
    Flatten, Dense, LeakyReLU, add
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Model

from trainer.config import checkpoint_dir, TRAIN_SPLIT, BATCH_SIZE, create_or_join


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


def make_or_restore_model(input_shape, model_type):
    """
    create or restore the model trained before
    :param model: keras model class
    :return: keras model class
    """
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoint = create_or_join('{}{}'.format(checkpoint_dir, model_type))
    checkpoints = [path.join(checkpoint, name) for name in os.listdir(checkpoint)]

    model = discard_model(input_shape) if model_type == 'discard' else rcpk_model(input_shape)
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring {} from".format(model_type), latest_checkpoint)
        model.load_weights(latest_checkpoint)
        return model
    print("Creating a new {} model".format(model_type))
    return model


def discard_model(input_shape):
    """
    Discard Model
    Network structure using idea of CNN
    :param input_shape: data shape
    :return: keras model class
    """
    x = input_shape
    x = Normalization()(x)
    for _ in range(3):
        x = Conv2D(256, (3, 1), padding="same", data_format="channels_last")(x)
    for _ in range(3):
        x = residual_block(x, 256, _project_shortcut=True)

    x = Conv2D(kernel_size=1, strides=1, filters=1, padding="same")(x)
    x = Flatten()(x)
    outputs = Dense(34, activation="softmax")(x)
    # model = keras.applications.ResNet50V2(weights=None, input_shape=(64, 34, 1), classes=34, include_top=True)
    model = Model(input_shape, outputs)
    model.summary()
    model.compile(
        keras.optimizers.Adam(learning_rate=0.008),
        keras.losses.CategoricalCrossentropy(),
        metrics=keras.metrics.CategoricalAccuracy())
    return model


def rcpk_model(input_shape):
    """
    Riichi, Chi, Pon, Kan models
    Network structure using idea of CNN
    :param input_shape: data shape
    :return: keras model class
    """
    x = input_shape
    x = Normalization()(x)
    for _ in range(3):
        x = Conv2D(256, (3, 1), padding="same", data_format="channels_last")(x)
    for _ in range(5):
        x = residual_block(x, 256, _project_shortcut=True)
    for _ in range(3):
        x = Conv2D(32, (3, 1), padding="same", data_format="channels_last")(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Dense(256)(x)
    outputs = Dense(2, activation="softmax")(x)

    model = Model(input_shape, outputs)
    model.summary()
    model.compile(
        keras.optimizers.Adam(learning_rate=0.008),
        keras.losses.BinaryCrossentropy(),
        metrics=keras.metrics.Accuracy())
    return model


def data_generator(states, labels, start=0):
    end = len(states) * TRAIN_SPLIT if start == 0 else len(states)
    while start < end:
        if start + BATCH_SIZE > end:
            yield states[start:].reshape((BATCH_SIZE, 4, 34, 1)), keras.utils.to_categorical(
                [i // 4 for i in labels[start:]], num_classes=34)
        else:
            yield states[start:start + BATCH_SIZE].reshape(
                (BATCH_SIZE, 4, 34, 1)), keras.utils.to_categorical(
                [i // 4 for i in labels[start:start + BATCH_SIZE]],
                num_classes=34)
        start += BATCH_SIZE


def DiscardFeatureGenerator(filepath, train_set=True):
    last_line = subprocess.check_output(['tail', '-1', filepath])
    total_element = json.loads(last_line)['id']
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            if not train_set and data['id'] < int(total_element * TRAIN_SPLIT) - 1:
                continue
            draw, hands, discard_pool, open_hands, label = data['draw_tile'], data['hands'], \
                                                           data['discarded_tiles_pool'], data[
                                                               'four_players_open_hands'], \
                                                           data['discarded_tile']
            hands_mat, draw_mat, discard_pool_mat, four_open_hands_mat = np.zeros((4, 34)), np.zeros((1, 34)), np.zeros(
                (4, 34)), np.zeros((4, 34))
            for tile in hands:
                hands_mat[tile % 4][tile // 4] = 1
            draw_mat[0][draw // 4] = 1
            for discard_tile in discard_pool:
                discard_pool_mat[discard_tile % 4][discard_tile // 4] = 1
            for player in open_hands:
                for tile in player:
                    four_open_hands_mat[tile % 4][tile // 4] = 1
            features = np.vstack((hands_mat, draw_mat, discard_pool_mat, four_open_hands_mat))
            yield features.reshape((features.shape[0], 34, 1)), keras.utils.to_categorical(label // 4, num_classes=34)
