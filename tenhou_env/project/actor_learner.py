import numpy as np
import tensorflow as tf
from tensorflow import keras
import ray
from tenhou.client import TenhouClient
from utils.logger import set_up_logging
from game.ai.configs.default import BotDefaultConfig
from game.ai.models import make_or_restore_model
from game.ai.utils import *
from keras import Input
from keras.optimizers import Adam

class Learner:
    def __init__(self, opt, model_type):
        self.opt = opt
        self.model_type = model_type
        
        self.actor = make_or_restore_model(input_shape_dict[model_type], model_type, "local")
        state_input = Input(self.actor.input.shape)
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(actor.output.shape))

        output = self.actor(state_input)
        self.model = keras.Model(inputs=[state_input, advantage, old_prediction], outputs=[output])
        self.model.compile(optimizer=Adam(lr=(LR)),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction = old_prediction
                      )])
        self.model.summary()        

    def get_weights(self):
        return self.actor.get_weights()

    def set_weights(self, weights):
        self.actor.set_weights(weights)

    def train(self, batch, cnt):
        feature, advantage, old_prediction, action = batch
        actor_loss = self.model.fit(x=[feature, advantage, old_prediction], y=[action], shuffle=True, epochs=EPOCHS, verbose=False)
        # writer
        # self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)  
        if cnt % 500 == 0:
            pass        #TO DO, do some summary writer thing


class Actor:
    def __init__(self, opt, job, buffer):
        self.opt = opt
        self.job = job
        self.bot_config = BotDefaultConfig()
        self.bot_config.buffer = buffer

    def set_weights(self, weights):
        self.bot_config.weights = weights       #a dict for all models

    def get_weights(self):
        pass

    def run(self):
        logger = set_up_logging()

        client = TenhouClient(logger, bot_config=self.bot_config)     

        for _ in range(self.opt.num_games):
            client.connect()

            try:
                was_auth = client.authenticate()

                if was_auth:
                    client.start_game()
                else:
                    client.end_game()
            except KeyboardInterrupt:
                logger.info("Ending the game...")
                client.end_game()
            except Exception as e:
                logger.exception("Unexpected exception", exc_info=e)
                logger.info("Ending the game...")
                client.end_game(False)  
                
        client.table.player.ai.write_buffer()
                  