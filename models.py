import datetime
import h5py
import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, \
    Flatten, Dense, LeakyReLU, add
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Model

BATCH_SIZE = 64
TRAIN_SPLIT = 0.8

checkpoint_dir = "./checkpoints"
if not os.path.exists('logs/stdout_logs'):
    os.makedirs('logs/stdout_logs')
sys.stderr = sys.stdout
sys.stdout = open('{}/{}.txt'.format('logs/stdout_logs', datetime.datetime.now().isoformat()), 'w')

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


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


# class BaseModel(keras.Model):
#     def __init__(self):
#         super(BaseModel, self).__init__()
#         self.conv1 = Conv2D(256, (3, 1), padding="same", data_format="channels_last")
#         self.conv2 = Conv2D(kernel_size=1, strides=1, filters=1, padding="same",)
#         self.outputs = Dense(34, activation="softmax")
#
#     def call(self, inputs):
#         x = inputs
#         x = Normalization()(x)
#         x = self.conv1(x)
#
#         for _ in range(5):
#             x = residual_block(x, 256, _project_shortcut=True)
#         x = self.conv2(x)
#         x = Flatten()(x)
#         x = self.outputs(x)
#         return x
def make_or_restore_model(model):
    """
    create or restore the model trained before
    :param model: keras model class
    :return: keras model class
    """
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
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
    for _ in range(5):
        x = residual_block(x, 256, _project_shortcut=True)

    x = Conv2D(kernel_size=1, strides=1, filters=1, padding="same")(x)
    x = Flatten()(x)
    outputs = Dense(34, activation="softmax")(x)
    return Model(input_shape, outputs)


def rcpk_model(input_shape):
    """
    Riichi, Chow, Pong, Kong models
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
    x = Dense(1024)(x)
    x = Dense(256)(x)
    outputs = Dense(34, activation="softmax")(x)
    return Model(input_shape, outputs)


def is_last_round():
    """
    Ending The Game
    A game typically finishes under one of two conditions:

    The final round of South (or, optionally, just East) is played
    One of the players goes below 0 points
    A fairly common but optional rule is that if no players are over 30,000 points by the end of South,
    then the game will continue into West round, and keeps going until any player gets above 30,000 points.

    It is worth noting that the conditions for ending the game will depend on the exact rules being played to.
    For example EMA Tournament rules will allow players to continue with negative points,
    and the game ends at the end of South round regardless of the score situation.
    :return:
    """


def is_lowest_score():
    """
    Check if has lowest_score
    :return:
    """


def winning_model(hands, game_stats, last_action):
    # check if this is the last round
    if is_last_round(game_stats):
        if is_lowest_score():
            return False
        else:
            return True


if __name__ == "__main__":
    # https://drive.google.com/uc\?id\=1iZpWSXRF9NlrLLwtxujLkAXk4k9KUgUN

    # all_files = glob.glob("./data" + "/*.csv")
    # if './data/2020.csv' in all_files:
    #     all_files.remove('./data/2020.csv')
    # if './data/None.csv' in all_files:
    #     all_files.remove('./data/None.csv')
    # li = np.array([])
    # for filename in all_files:
    #     print(filename)
    #     df = pd.read_csv(filename, engine='python', error_bad_lines=False)
    #     s = preprocess(df['log_content'].dropna()).values.tolist()
    #     # li = np.concatenate((li, df['log_content'].dropna()), axis=0)

    # df = pd.read_csv('./data/2021.csv', engine='python', error_bad_lines=False)
    # li = df['log_content'].dropna().values.tolist()
    # states, labels = preprocess(li)
    f = h5py.File('logs_parser/discarded_model_dataset_sum_2021.hdf5', 'r')
    states = f.get('hands')
    labels = f.get('discarded_tile')
    print(states.shape)


    # states = states.reshape((len(states), 4, 34, 1))

    def data_generator(data, start, label=False):
        end = len(data) * TRAIN_SPLIT if start == 0 else len(data)
        while start < end:
            if start + BATCH_SIZE > end:
                yield data[start:].reshape((BATCH_SIZE, 4, 34, 1)) if not label else keras.utils.to_categorical(
                    [i // 4 for i in data[start:]], num_classes=34)
            else:
                yield data[start:start + BATCH_SIZE].reshape(
                    (BATCH_SIZE, 4, 34, 1)) if not label else keras.utils.to_categorical(
                    [i // 4 for i in data[start:start + BATCH_SIZE]],
                    num_classes=34)
            start += BATCH_SIZE


    def train_generator(states, labels):
        i = 0
        train_size = len(states) * TRAIN_SPLIT
        while i < train_size:
            if i + BATCH_SIZE > train_size:
                yield states[i:].reshape((BATCH_SIZE, 4, 34, 1)), keras.utils.to_categorical(
                    [i // 4 for i in labels[i:]], num_classes=34)
            else:
                yield states[i:i + BATCH_SIZE].reshape((BATCH_SIZE, 4, 34, 1)), keras.utils.to_categorical(
                    [i // 4 for i in labels[i:i + BATCH_SIZE]], num_classes=34)
            i += BATCH_SIZE


    def validation_generator(states, labels):
        i = len(states) * TRAIN_SPLIT
        while i < len(states):
            if i + BATCH_SIZE > len(states):
                yield states[i:].reshape((BATCH_SIZE, 4, 34, 1)), keras.utils.to_categorical(
                    [i // 4 for i in labels[i:]], num_classes=34)
            else:
                yield states[i:i + BATCH_SIZE].reshape((BATCH_SIZE, 4, 34, 1)), keras.utils.to_categorical(
                    [i // 4 for i in labels[i:i + BATCH_SIZE]], num_classes=34)
            i += BATCH_SIZE


    train_dataset = tf.data.Dataset.from_generator(lambda: train_generator(states, labels), (tf.int8, tf.int8))
    val_dataset = tf.data.Dataset.from_generator(lambda: validation_generator(states, labels), (tf.int8, tf.int8))

    input_shape = keras.Input((4, 34, 1))
    model = make_or_restore_model(discard_model(input_shape))

    callbacks = [
        keras.callbacks.TensorBoard(log_dir="./logs", update_freq='epoch', histogram_freq=1),
        keras.callbacks.ModelCheckpoint('./checkpoints/',
                                        save_weights_only=True, monitor='val_accuracy',
                                        save_freq="epoch",
                                        mode='max', save_best_only=True),
        keras.callbacks.EarlyStopping(),
        keras.callbacks.LearningRateScheduler(scheduler)
    ]

    model.summary()
    model.compile(
        keras.optimizers.Adam(learning_rate=0.008),
        keras.losses.CategoricalCrossentropy(),
        metrics=keras.metrics.CategoricalAccuracy())

    model.fit(train_dataset, epochs=50, batch_size=BATCH_SIZE, validation_data=val_dataset, use_multiprocessing=True,
              workers=-1,
              callbacks=callbacks)
    sys.stdout.close()
