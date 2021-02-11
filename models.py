import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from logs_parser.parser import parse_mjlog
from logs_parser.viewer import print_node
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import glob, os, sys, datetime

if not os.path.exists('logs/stdout_logs'):
    os.mkdir('logs/stdout_logs')
sys.stderr = sys.stdout
sys.stdout = open('{}/{}.txt'.format('logs/stdout_logs', datetime.datetime.now().isoformat()), 'w')


def preprocess(df_data):
    states = []
    label = []
    for i, game in enumerate(df_data):
        node = ET.fromstringlist(game)
        data = parse_mjlog(node)
        for round in data['rounds']:
            players_hands = round[0]['data']['hands']
            encoded_hands = np.zeros((4, 4, 34))
            for j, hand in enumerate(players_hands):
                hand.sort()
                for t in hand:
                    encoded_hands[j][t % 4][t // 4] = 1

            for action in round[1:]:
                if action['tag'] not in ["DRAW", "DISCARD"]:
                    continue
                player = action['data']['player']
                same_tile_pos = action["data"]['tile'] % 4
                tile = action["data"]['tile'] // 4
                if action['tag'] == "DRAW":
                    encoded_hands[player][same_tile_pos][tile] = 1
                elif action['tag'] == "DISCARD":
                    if np.count_nonzero(encoded_hands[player]) == 14:
                        states.append(encoded_hands[player].copy())
                        label.append(tile)
                    encoded_hands[player][same_tile_pos][tile] = 0
    return states, label

def relu_bn(inputs):
    relu =ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn
def residual_block(x, filters, kernel_size, downsample):
    y = Conv2D(kernel_size=kernel_size, strides= 1 if not downsample else 2, filters=filters, padding="same", data_format="channels_last")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size, strides= 1, filters=filters, padding="same", data_format="channels_last")(y)
    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same", data_format="channels_last")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

class BaseModel(keras.Model):
    def __init__(self, in_shape):
        super(BaseModel, self).__init__()
        self.in_shape = in_shape
        self.conv1 = Conv2D(256, (3, 1), padding="same", data_format="channels_last")
        self.conv2 = Conv2D(kernel_size=1, strides=1, filters=1, padding="same", data_format="channels_last")
        self.outputs = keras.layers.Dense(34, activation="softmax")

    def residual_block(self, x, filters, kernel_size, downsample):
        y = Conv2D(kernel_size=kernel_size, strides= 1 if not downsample else 2, filters=filters, padding="same")(x)
        y = relu_bn(y)
        y = Conv2D(kernel_size=kernel_size, strides= 1, filters=filters, padding="same")(y)
        if downsample:
            x = Conv2D(kernel_size=1,
                       strides=2,
                       filters=filters,
                       padding="same")(x)
        out = Add()([x, y])
        out = relu_bn(out)
        return out

    def call(self, inputs, **kwargs):
        x = self.in_shape
        for _ in range(3):
            x = self.conv1(x)
        x = self.residual_block(x, 256, (3, 1), False)
        x = self.conv2(x)
        x = keras.layers.Flatten()(x)
        x = self.outputs(x)
        return keras.models.Model(self.in_shape, x)


def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = keras.layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.LeakyReLU()(y)

    y = keras.layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = keras.layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = keras.layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = keras.layers.BatchNormalization()(shortcut)

    y = keras.layers.add([shortcut, y])
    y = keras.layers.LeakyReLU()(y)

    return y


if __name__ == "__main__":
    # https://drive.google.com/uc\?id\=1iZpWSXRF9NlrLLwtxujLkAXk4k9KUgUN

    all_files = glob.glob("./data" + "/*.csv")
    if './data/2020.csv' in all_files:
        all_files.remove('./data/2020.csv')
    if './data/None.csv' in all_files:
        all_files.remove('./data/None.csv')
    li = np.array([])
    for filename in all_files:
        df = pd.read_csv(filename, engine='python', error_bad_lines=False)
        li = np.concatenate((li, df['log_content'].dropna()), axis=0)



    # df = pd.read_csv('./data/2021.csv', engine='python', error_bad_lines=False)
    # li = df['log_content']

    print(li.shape)
    states, label = preprocess(li)
    states = np.array(states)

    states = states.reshape((len(states), 4, 34, 1))
    label = keras.utils.to_categorical(label, num_classes=34)
    train = int(len(states)*0.8)
    x_train = states[:train]
    x_label = label[:train]
    valid_data = states[train:]
    valid_label = label[train:]

    input_shape = keras.Input((4, 34, 1))
    x = input_shape
    # x = x_train
    x = keras.layers.experimental.preprocessing.Normalization()(x)
    x = Conv2D(256, (3, 1), padding="same", data_format="channels_last")(x)
    x = Conv2D(256, (3, 1), padding="same", data_format="channels_last")(x)
    x = Conv2D(256, (3, 1), padding="same", data_format="channels_last")(x)
    for _ in range(5):
        x = residual_block(x, 256, _project_shortcut=True)

    x = Conv2D(kernel_size=1, strides=1, filters=1, padding="same")(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(34, activation="softmax")(x)
    # m = keras.applications.resnet.ResNet50(
    #     include_top=False,
    #     weights=None,
    #     input_shape=(4, 34, 1),
    #     pooling=None,
    #     classes=34,
    #     # classifier_activation="softmax",
    # )
    # x = m.output
    # x = keras.layers.Flatten()(x)
    # outputs = keras.layers.Dense(34, activation="softmax")(x)
    model = keras.models.Model(input_shape, outputs)
    # model = BaseModel(input_shape)
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    callbacks = [
        keras.callbacks.TensorBoard(log_dir="./logs", update_freq='epoch', histogram_freq=1),
        keras.callbacks.ModelCheckpoint('./checkpoints/',
                                        save_weights_only=True, monitor='val_accuracy',
                                        save_freq="epoch",
                                        mode='max', save_best_only=True),
        keras.callbacks.EarlyStopping(),
        keras.callbacks.LearningRateScheduler(scheduler)
    ]

    # model.summary()
    model.compile(
        keras.optimizers.Adam(learning_rate=0.008),
        keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"])
    model.fit(x_train, x_label, epochs=50, batch_size=64, validation_data=(valid_data, valid_label),
              callbacks=callbacks)
    sys.stdout.close()


