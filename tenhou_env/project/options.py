import numpy as np
import os
import sys
import datetime
from math import ceil


class Options:
    def __init__(self, num_nodes, num_workers):
        # parameters set

        self.num_nodes = num_nodes
        self.num_workers = num_workers
        self.num_games = 10
        self.num_learners = 1

        self.push_freq = 100

        self.gamma = 0.99

        # self.a_l_ratio = a_l_ratio
        # self.weights_file = weights_file

        self.recover = False
        self.checkpoint_freq = 21600  # 21600s = 6h

        # gpu memory fraction
        self.gpu_fraction = 0.3

        self.hidden_size = [400, 300]


        self.buffer_size = int(1e6)
        self.buffer_size = self.buffer_size // self.num_buffers

        # self.start_steps = int(1e4) // self.num_buffers

        # if self.weights_file:
        #     self.start_steps = self.buffer_size

        self.lr = 1e-3
        self.polyak = 0.995

        self.batch_size = 128

        # n-step
        self.Ln = 1

        self.save_freq = 1

        self.seed = 0

        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.summary_dir = ROOT_DIR + '/tboard_ray'  # Directory for storing tensorboard summary results
        self.save_dir = ROOT_DIR   # Directory for storing trained model
        self.save_interval = int(5e5)

        self.log_dir = self.summary_dir + "/" + str(datetime.datetime.now()) + "-workers_num:" + \
                       str(self.num_workers)
