import numpy as np
import tensorflow as tf
from tensorflow import keras
import ray
from tenhou.client import TenhouClient
from utils.logger import set_up_logging
from game.ai.configs.default import BotDefaultConfig

class Learner:
    def __init__(self, opt, model_type):
        self.opt = opt
        self.model_type = model_type
        self.model = None

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def train(self, batch, cnt):
        self.model.fit(batch[0], batch[1])
        if cnt % 500 == 0:
            pass        #TO DO, do some summary writer thing


class Actor:
    def __init__(self, opt, job):
        self.opt = opt
        self.job = job
        self.bot_config = BotDefaultConfig()

    def set_weights(self, model_type, weights):
        self.bot_config.weights = {model_type: weights}

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