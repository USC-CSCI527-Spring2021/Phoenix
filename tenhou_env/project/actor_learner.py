import numpy as np
import tensorflow as tf
from tensorflow import keras
import ray
from tenhou.client import TenhouClient
from utils.logger import set_up_logging

class Actor:
    def __init__(self, opt, job):
        self.opt = opt
        self.job = job
    
    def set_weights(self, model, weights):
        pass

    def get_weights(self):
        pass

    def run(self):
        for _ in range(self.opt.num_games):
            logger = set_up_logging()

            client = TenhouClient(logger)
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