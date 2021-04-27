import importlib
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import numpy as np


from bots_battle import _set_up_bots_battle_game_logger, main as bot_battle_main
from game.ai.configs.default import BotDefaultConfig
from game.ai.utils import *
from game.bots_battle.game_manager import GameManager
from game.bots_battle.local_client import LocalClient
from tenhou.client import TenhouClient
from utils.logger import set_up_logging
from utils.settings_handler import settings
import ray

class Learner:
    def __init__(self, opt, model_type):
        from tensorflow.python.framework.ops import disable_eager_execution
        disable_eager_execution()
        self.opt = opt
        self.model_type = model_type

        self.actor = keras.models.load_model(os.path.join(os.getcwd(), 'models', model_type))

        state_input = keras.Input(self.actor.input.shape[1:])
        advantage = keras.Input(shape=(1,))
        old_prediction = keras.Input(shape=self.actor.output.shape[1:])
        


        output = self.actor(state_input)
        self.model = keras.Model(inputs=[state_input, advantage, old_prediction], outputs=[output])
        self.model.compile(optimizer=Adam(learning_rate=LR),
                           loss=[proximal_policy_optimization_loss(
                               advantage=advantage,
                               old_prediction=old_prediction
                           )],
                           metrics=[tf.keras.metrics.Accuracy(name="accuracy")],
                           experimental_run_tf_function=False)

    def get_weights(self):
        return self.actor.get_weights()

    def set_weights(self, weights):
        self.actor.set_weights(weights)

    def train(self, batch, cnt):
        feature, advantage, old_prediction, action = zip(*batch)

        feature = np.asarray(feature)
        advantage = np.asarray(advantage)
        old_prediction = np.asarray(old_prediction)
        action = np.asarray(action)
        print(f"batch shape: feature:{feature.shape}, advantage:{advantage.shape}")
        hist = self.model.fit(x=[np.asarray(feature), np.asarray(advantage), np.asarray(old_prediction)], y=[np.asarray(action)], shuffle=True, epochs=EPOCHS,
                                    verbose=False)
        print(f"epoch: {cnt}, actor loss: {sum(hist.loss) / len(hist.loss)}")
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
        self.bot_config.weights = weights  # a dict for all models

    def get_weights(self):
        pass

    def run(self):
        assert self.bot_config.weights, "should set weights before run the actor"
        assert len(self.bot_config.weights) == 6, "Require 6 model weights"
        # for later use in nn.py
        self.bot_config.isOnline = self.opt.isOnline
        if self.opt.isOnline:
            module = importlib.import_module(f"settings.base")
            for key, value in vars(module).items():
                # let's use only upper case settings
                if key.isupper():
                    settings.__setattr__(key, value)

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
        else:
            _set_up_bots_battle_game_logger()        

            print_logs = True

            clients = []
            replay_name = GameManager.generate_replay_name()
            for i in range(4):
                self.bot_config.name = f"bot{i}"
                clients.append(LocalClient(self.bot_config, print_logs, replay_name, i))
                

            bot_battle_main(self.opt.num_games, print_logs, clients, replay_name)
                
            # Write to buffer
            # for one_game_clients in clients:
            print('start write buffer')
            # for client in clients:
            print(ray.get([b.get_counts.remote() for b in self.bot_config.buffer.values()]))
            clients[0].table.player.ai.write_buffer()
