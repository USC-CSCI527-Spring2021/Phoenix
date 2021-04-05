import ray
import numpy as np
import os
import sys
import multiprocessing
import json
import copy

@ray.remote
class ReplayBuffer:
    def __init__(self, opt, buffer_index, buffer_type):
        self.opt = opt
        # self.buffer_index = buffer_index
        self.buffer_type = buffer_type
        # self.obs1_buf = np.zeros([opt.buffer_size, opt.obs_dim], dtype=np.float32)
        # self.obs2_buf = np.zeros([opt.buffer_size, opt.obs_dim], dtype=np.float32)
        # self.acts_buf = np.zeros(opt.buffer_size, dtype=np.float32)
        # self.rews_buf = np.zeros(opt.buffer_size, dtype=np.float32)
        self.buf = []
        self.ptr, self.size, self.max_size = 0, 0, opt.buffer_size
        self.actor_steps, self.learner_steps = 0, 0
        
    def store(self, obs, act, rew, next_obs):
        # self.obs1_buf[self.ptr] = obs
        # self.obs2_buf[self.ptr] = next_obs
        # self.acts_buf[self.ptr] = act
        # self.rews_buf[self.ptr] = rew
        self.buf[self.ptr] = [obs, act, rew, next_obs]
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.actor_steps += 1

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.opt.batch_size)
        self.learner_steps += 1
        # return dict(obs1=self.obs1_buf[idxs],
        #             obs2=self.obs2_buf[idxs],
        #             acts=self.acts_buf[idxs],
        #             rews=self.rews_buf[idxs])
        return self.buf[idxs]
    
    def get_counts(self):
        return self.learner_steps, self.actor_steps, self.size
    
    def save(self):
        info = {'buffer': self.buf,
                'ptr': self.ptr,
                'size': self.size,
                'max_size': self.max_size,
                'learner_steps': self.learner_steps,
                'actor_steps': self.actor_steps}
        np.save(self.opt.save_dir + '/' + self.buffer_type, info)
        print("**** buffer " + self.buffer_type + " saved! *******")
    
    def load(self, buffer_path):
        if not buffer_path:
            buffer_path = self.opt.save_dir + '/' + self.buffer_type + '.npy'
        info = np.load(buffer_path)
        self.buf, self.ptr, self.size, self.max_size, self.learner_steps, self.actor_steps = info['buffer'],
                 info['ptr'], info['size'], info['max_size'], info['learner_steps'], info['actor_steps']
        print("****** buffer " + self.buffer_type + " restored! ******")
        print("****** buffer " + self.buffer_type + " infos:", self.ptr, self.size, self.max_size,
              self.actor_steps, self.learner_steps)


@ray.remote
class ParameterServer:
    def __init__(self, opt, weights_file, checkpoint_path, ps_index):
        # each node will have a Parameter Server

        self.opt = opt
        self.learner_step = 0
        net = Learner(opt, job="ps")
        keys, values = net.get_weights()

        # --- make dir for all nodes and save parameters ---
        try:
            os.makedirs(opt.save_dir)
            os.makedirs(opt.save_dir + '/checkpoint')
        except OSError:
            pass
        all_parameters = copy.deepcopy(vars(opt))
        all_parameters["obs_space"] = ""
        all_parameters["act_space"] = ""
        with open(opt.save_dir + "/" + 'All_Parameters.json', 'w') as fp:
            json.dump(all_parameters, fp, indent=4, sort_keys=True)
        # --- end ---

        self.weights = None

        if not checkpoint_path:
            checkpoint_path = opt.save_dir + "/checkpoint"

        if opt.recover:
            with open(checkpoint_path + "/checkpoint_weights.pickle", "rb") as pickle_in:
                self.weights = pickle.load(pickle_in)
                print("****** weights restored! ******")

        if weights_file:
            try:
                with open(weights_file, "rb") as pickle_in:
                    self.weights = pickle.load(pickle_in)
                    print("****** weights restored! ******")
            except:
                print("------------------------------------------------")
                print(weights_file)
                print("------ error: weights file doesn't exist! ------")
                exit()

        if not opt.recover and not weights_file:
            values = [value.copy() for value in values]
            self.weights = dict(zip(keys, values))

    def push(self, keys, values):
        values = [value.copy() for value in values]
        for key, value in zip(keys, values):
            self.weights[key] = value
        self.learner_step += opt.push_freq

    def pull(self, keys):
        return [self.weights[key] for key in keys]

    def get_weights(self):
        return copy.deepcopy(self.weights)

    # save weights to disk
    def save_weights(self):
        with open(self.opt.save_dir + "/checkpoint/" + "checkpoint_weights.pickle", "wb") as pickle_out:
            pickle.dump(self.weights, pickle_out)

@ray.remote(num_cpus=2, num_gpus=1, max_calls=1)    #centralized training
def worker_train(ps, node_buffer, opt, learner_index):
    
