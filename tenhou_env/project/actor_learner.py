import importlib
import os

from tensorflow import keras

from bots_battle import _set_up_bots_battle_game_logger, main as bot_battle_main
from game.ai.configs.default import BotDefaultConfig
from game.ai.utils import *
from game.bots_battle.game_manager import GameManager
from game.bots_battle.local_client import LocalClient
from tenhou.client import TenhouClient
from utils.logger import set_up_logging
from utils.settings_handler import settings


class Learner:
    def __init__(self, opt, model_type):
        self.opt = opt
        self.model_type = model_type

        self.actor = keras.models.load_model(os.path.join(os.getcwd(), 'models', model_type))

        # state_input = self.actor.input
        # advantage = keras.Input(shape=(1,))
        # old_prediction = self.actor.output
        #
        # output = self.actor(state_input)
        # self.model = keras.Model(inputs=[state_input, advantage, old_prediction], outputs=[output])
        # self.model.compile(optimizer=Adam(lr=(LR)),
        #                    loss=[proximal_policy_optimization_loss(
        #                        advantage=advantage,
        #                        old_prediction=old_prediction
        #                    )])
        # self.model.summary()

    def get_weights(self):
        return self.actor.get_weights()

    def set_weights(self, weights):
        self.actor.set_weights(weights)

    def train(self, batch, cnt):
        feature, advantage, old_prediction, action = batch
        actor_loss = self.actor.fit(x=[feature, advantage, old_prediction], y=[action], shuffle=True, epochs=EPOCHS,
                                    verbose=False)
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
        self.player_idx = 0

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
            module = importlib.import_module(f"settings.bot_{self.player_idx}_settings")
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
                print("One local game start!")
                bot_battle_main(self.opt.num_games, print_logs, clients, replay_name)
                print("One local game end!")
                # end of a local game
                # clients.append(one_game_clients)
            # Write to buffer
            for client in clients:
                client.table.player.ai.write_buffer()
