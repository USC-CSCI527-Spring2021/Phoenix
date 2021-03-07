import os
from os import path

import numpy as np
import tensorflow as tf
from google.cloud import storage
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, \
    Flatten, Dense, LeakyReLU, add
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Model

from trainer.config import CHECKPOINT_DIR, TRAIN_SPLIT, BATCH_SIZE, create_or_join, RANDOM_SEED, GCP_BUCKET

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


def make_or_restore_model(input_shape, model_type):
    """
    create or restore the model trained before
    :param model: keras model class
    :return: keras model class
    """
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    is_cloud = bool(os.environ.get("TF_KERAS_RUNNING_REMOTELY"))

    checkpoint = create_or_join('{}/{}'.format(CHECKPOINT_DIR, model_type))

    if is_cloud:
        client = storage.Client()
        checkpoints = [path.join(checkpoint, name) for name in list(client.list_blobs(GCP_BUCKET, prefix=checkpoint))]
    else:
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


def split_dataset(dataset: tf.data.Dataset):
    """
    Splits a dataset of type tf.data.Dataset into a training and validation dataset using given ratio. Fractions are
    rounded up to two decimal places.
    @param dataset: the input dataset to split.
    @return: a tuple of two tf.data.Datasets as (training, validation)
    """
    validation_data_fraction = 1 - TRAIN_SPLIT
    validation_data_percent = round(validation_data_fraction * 100)
    if not (0 <= validation_data_percent <= 100):
        raise ValueError("validation data fraction must be ∈ [0,1]")

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
    validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    validation_dataset = validation_dataset.map(lambda f, data: data)

    return train_dataset, validation_dataset


# def read


def transform_discard_features(data):
    """
    Transform discard raw data to input features and labels
    :param data: each row of discard raw data
    :return: features and labels
    """
    # res = {
    #     u'draw_tile': str(draw_tile_list[k]),
    #     u'hands': str(hands_list[k]),
    #     u'discarded_tiles_pool': str(discarded_tiles_pool_list[k]),
    #     u'four_players_open_hands': str(four_players_open_hands_list[k]),
    #     u'discarded_tile': discarded_tile[k]
    # }
    # out_features, out_labels = [], []
    # for data in batch_data:
    #     draw, hands, discard_pool, open_hands, label = data['draw_tile'], data['hands'], \
    #                                                    data['discarded_tiles_pool'], data[
    #                                                        'four_players_open_hands'], \
    #                                                    data['discarded_tile']
    #     hands_mat, draw_mat, discard_pool_mat, four_open_hands_mat = np.zeros((4, 34)), np.zeros((1, 34)), np.zeros(
    #         (4, 34)), np.zeros((4, 34))
    #     for tile in hands:
    #         hands_mat[tile % 4][tile // 4] = 1
    #     draw_mat[0][draw // 4] = 1
    #     for discard_tile in discard_pool:
    #         discard_pool_mat[discard_tile % 4][discard_tile // 4] = 1
    #     for player in open_hands:
    #         for tile in player:
    #             four_open_hands_mat[tile % 4][tile // 4] = 1
    #     features = np.vstack((hands_mat, draw_mat, discard_pool_mat, four_open_hands_mat))
    #     out_features.append(features.reshape((features.shape[0], 34, 1)))
    #     out_labels.append(keras.utils.to_categorical(label // 4, num_classes=34))
    # return out_features, out_labels
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
    return {"features": features.reshape((features.shape[0], 34, 1)),
            "labels": keras.utils.to_categorical(int(label // 4), 34)}
    # return [features.reshape((features.shape[0], 34, 1)), label // 4]
