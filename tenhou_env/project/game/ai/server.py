from tensorflow import keras
import os
import numpy as np
from keras.layers import Input
from game.ai.utils import proximal_policy_optimization_loss
from keras.optimizers import Adam
import tensorflow as tf
import datetime
'''
TO DO: implement Advantage as Q - V
'''
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 3
def buffer_reader(buffer_path):
    # TO DO: implement replay buffer select logic
    with open(buffer_path) as buffer:
        while True:
            yield [np.asarray(x) for x in zip(*np.random.choice(buffer, size=BATCH_SIZE))]

class PGTrainer:
    def __init__(self, model_type):
        self.model_path = os.path.join(os.getcwd(), 'models', model_type)
        self.buffer_path = os.path.join(os.getcwd(), 'buffer', model_type)
        

    def build_model(self):
        actor = keras.models.load_model(self.model_path)
        
        state_input = Input(actor.input.shape)
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(actor.output.shape))

        output = actor(state_input)
        model = keras.Model(inputs=[state_input, advantage, old_prediction], outputs=[output])
        model.compile(optimizer=Adam(lr=(LR)),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction = old_prediction
                      )])
        model.summary()
        return model

    def run(self):
        self.model = self.build_model()
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        generator = buffer_reader(self.buffer_path)

        for _ in range(EPOCHS):
            features, action, importance, advantage = next(generator)
            self.model.fit([features, advantage, importance], [action], callbacks=[tensorboard_callback])
