import copy
import glob
import multiprocessing
import os
import pickle
import time

import keras
import numpy as np
import ray
import tensorflow as tf
import sys

from actor_learner import Learner, Actor
from options import Options
from game.ai.utils import model_types

flags = tf.compat.v1.flags
FLAGS = tf.compat.v1.flags.FLAGS
flags.DEFINE_integer("num_nodes", 1, "number of nodes")
flags.DEFINE_integer("num_workers", 1, "number of workers")


@ray.remote
class ReplayBuffer:
    def __init__(self, opt, buffer_index, buffer_type):
        self.opt = opt
        self.buffer_type = buffer_type
        self.buffer_index = buffer_index
        self.ptr, self.size, self.max_size = 0, 0, opt.buffer_size
        self.buf = np.array([[]] * self.max_size)
        self.actor_steps, self.learner_steps = 0, 0
        #self.load()

    def store(self, obs, rew, pred, act):
        self.buf[self.ptr] = [obs, rew, pred, act]
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.actor_steps += 1

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.opt.batch_size)
        self.learner_steps += 1
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

        buffer_save_folder = self.opt.save_dir + f'/buffer/{str(self.buffer_index)}/'

        with open(buffer_save_folder+f"{self.buffer_type}.pkl", 'wb') as f:
            pickle.dump(info, f)
        print(f"**** buffer{self.buffer_index}" + self.buffer_type + " saved! *******")

    def load(self, buffer_path=None):

        if not buffer_path:
            buffer_path = self.opt.save_dir + f'/buffer/{str(self.buffer_index)}/' + self.buffer_type + '.pkl'
        info = pickle.load(open(buffer_path, 'rb'))
        self.buf, self.ptr, self.size, self.max_size, self.learner_steps, self.actor_steps = np.asarray(info['buffer']), info['ptr'], info['size'], info['max_size'], info['learner_steps'], info['actor_steps']
        print(f"****** buffer{self.buffer_index} " + self.buffer_type + " restored! ******")
        print(f"****** buffer{self.buffer_index} " + self.buffer_type + " infos:", self.ptr, self.size, self.max_size,
              self.actor_steps, self.learner_steps)


class Cache():

    def __init__(self, node_buffer):
        # cache for training data and model weights
        print('os.pid:', os.getpid())
        self.node_buffer = node_buffer
        self.q1 = {k: multiprocessing.Queue(12) for k in model_types}       #store sample batch
        self.q2 = multiprocessing.Queue(3)                                  #store weights
        self.p1 = multiprocessing.Process(target=self.ps_update, args=(self.q1, self.q2, self.node_buffer))
        self.p1.daemon = True

    def ps_update(self, q1, q2, node_buffer):
        print('os.pid of put_data():', os.getpid())

        node_idx = np.random.choice(opt.num_nodes, 1)[0]
        for model_type in model_types:
            q1[model_type].put(copy.deepcopy(ray.get(node_buffer[node_idx][model_type].sample_batch.remote())))
            print(f"**** fetched successfully, size {q1[model_type].qsize()}")
        # while True:
        for model_type in model_types:
            if q1[model_type].qsize() < 10:
                node_idx = np.random.choice(opt.num_nodes, 1)[0]
                q1[model_type].put(copy.deepcopy(ray.get(node_buffer[node_idx][model_type].sample_batch.remote())))
                print(f"**** fetched successfully, size {q1[model_type].qsize()}")

        if not q2.empty():
            keys, values = q2.get()
            [node_ps[i].push.remote(keys, values) for i in range(opt.num_nodes)]

    def start(self):
        #self.ps_update(self.q1, self.q2, self.node_buffer)
        # self.p1.start()
        # self.p1.join(15)
        print(f"#######******* size of qsize {[self.q1[model_type].qsize() for model_type in model_types]} ******#####")

    def end(self):
        self.p1.terminate()


@ray.remote
class ParameterServer:
    """
    One PS contains 1 or 4 clients weights
    """

    def __init__(self, opt, ps_index, weights_files=None, checkpoint_path=None):
        # each node will have a Parameter Server

        self.opt = opt
        self.learner_step = 0
        self.ps_index = ps_index
        self.weights = {}
        # --- make dir for all nodes and save parameters ---
        # try:
        #     os.makedirs(opt.save_dir)
        #     os.makedirs(opt.save_dir + '/checkpoint')
        # except OSError:
        #     pass
        # all_parameters = copy.deepcopy(vars(opt))
        # all_parameters["obs_space"] = ""
        # all_parameters["act_space"] = ""
        # with open(opt.save_dir + "/" + 'All_Parameters.json', 'w') as fp:
        # json.dump(all_parameters, fp, indent=4, sort_keys=True)
        # --- end ---

        if not checkpoint_path:
            checkpoint_path = opt.save_dir + "/checkpoint"

        # if opt.recover:
        #     with open(checkpoint_path + "/checkpoint_weights.pickle", "rb") as pickle_in:
        #         self.weights = pickle.load(pickle_in)
        #         print("****** weights restored! ******")

        if weights_files:
            assert len(weights_files) == 6, f"only {len(weights_files)} model files, need 6"
            print("****** ps start loading weights ******")
            try:
                for f in weights_files:
                    model_type = f.split("/")[-1]
                    weights = keras.models.load_model(f).get_weights()
                    self.weights[model_type] = weights
                    # with open(f, "rb") as pickle_in:
                    #     self.weights[f.split("/")[-1]] = pickle.load(pickle_in)
                print("****** weights restored! ******")
            except:
                print("------------------------------------------------")
                print(weights_files)
                print("------ error: weights file doesn't exist! ------")
                exit()

        # if not opt.recover and not weights_file:
        #     values = [value.copy() for value in values]
        #     self.weights = dict(zip(keys, values))

    def push(self, model_type, weights):
        self.weights[model_type] = weights
        self.learner_step += self.opt.push_freq

    def pull(self, model_type=None):
        if model_type:
            return self.weights[model_type]
        else:
            return self.weights

    def get_weights(self):
        return copy.deepcopy(self.weights)

    # save weights to disk
    def save_weights(self):
        with open(self.opt.save_dir + "/checkpoint/" + "checkpoint_weights.pickle", "wb") as pickle_out:
            pickle.dump(self.weights, pickle_out)
            print("******* PS saved successfully ********")


@ray.remote(num_cpus=1, num_gpus=0, max_calls=1)  # centralized training
def worker_train(ps, node_buffer, opt):
    from actor_learner import Learner, Actor
    from options import Options

    agents = {}
    for model_type in model_types:
        agent = Learner(opt, model_type)
        weights = ray.get(ps.pull.remote(model_type))
        agent.set_weights(weights)
        agents[model_type] = agent

    cache = Cache(node_buffer)
    cache.start()

    cnt = 1
    while True:
        for model_type in model_types:
            if cache.q1[model_type].empty():
                continue
            batch = cache.q1[model_type].get()
            print(f" ******* get batch of size {len(batch)} for model {model_type} *********")
            agents[model_type].train(batch, cnt)
            print('one batch trained')
        if cnt % opt.push_freq == 0:
            cache.q2.put(agent.get_weights())
        cnt += 1


@ray.remote
def worker_rollout(ps, replay_buffer, opt):

    from actor_learner import Learner, Actor
    from options import Options
    agent = Actor(opt, job='worker', buffer=replay_buffer)
    while True:
        weights = ray.get(ps.pull.remote())
        agent.set_weights(weights)
        agent.run()
        print("******* rollout agent finished a game ******")


@ray.remote
def worker_test(ps, node_buffer, opt):
    from actor_learner import Learner, Actor
    from options import Options    
    agent = Actor(opt, job="test", buffer=node_buffer[0])      
    init_time = time.time()
    save_times = 0
    checkpoint_times = 0

    while True:
        weights = ray.get(ps.get_weights.remote())
        agent.set_weights(weights)
        start_actor_step, start_learner_step, _ = get_al_status(node_buffer)
        start_time = time.time()

        agent.run()
        print("****** Test agent finished a game *******")
        last_actor_step, last_learner_step, _ = get_al_status(node_buffer)
        actor_step = np.sum(last_actor_step) - np.sum(start_actor_step)
        learner_step = np.sum(last_learner_step) - np.sum(start_learner_step)
        alratio = actor_step / (learner_step + 1)
        update_frequency = int(learner_step / (time.time() - start_time))
        total_learner_step = np.sum(last_learner_step)

        print("---------------------------------------------------")
        print("frame freq:", np.round((last_actor_step - start_actor_step) / (time.time() - start_time)))
        print("actor_steps:", np.sum(last_actor_step), "learner_step:", total_learner_step)
        print("actor leaner ratio: %.2f" % alratio)
        print("learner freq:", update_frequency)
        print("Ray total resources:", ray.cluster_resources())
        print("available resources:", ray.available_resources())
        print("---------------------------------------------------")

        buffer_save_op = [node_buffer[node_index][model_type].save.remote() for model_type in model_types for
                            node_index in range(opt.num_nodes)]
        ray.wait(buffer_save_op, num_returns=opt.num_nodes*5)
        print("saved successfully!!!!!")

        total_time = time.time() - init_time

        if total_learner_step // opt.save_interval > save_times:
            with open(opt.save_dir + "/" + str(total_learner_step / 1e6) + "_weights.pickle", "wb") as pickle_out:
                pickle.dump(weights, pickle_out)
                print("****** Weights saved by time! ******")
            save_times = total_learner_step // opt.save_interval

        # save everything every checkpoint_freq s
        if total_time // opt.checkpoint_freq > checkpoint_times:
            print("save everything!")
            save_start_time = time.time()

            ps_save_op = [node_ps[i].save_weights.remote() for i in range(opt.num_nodes)]
            buffer_save_op = [node_buffer[node_index][model_type].save.remote() for model_type in model_types for
                              node_index in range(opt.num_nodes)]
            ray.wait(buffer_save_op + ps_save_op, num_returns=opt.num_nodes * 6)  # 5 models + ps

            print("total time for saving :", time.time() - save_start_time)
            checkpoint_times = total_time // opt.checkpoint_freq



def get_al_status(node_buffer):
    buffer_learner_step = []
    buffer_actor_step = []
    buffer_cur_size = []

    for node_index in range(opt.num_nodes):
        for model_type in model_types:
            learner_step, actor_step, cur_size = ray.get(node_buffer[node_index][model_type].get_counts.remote())
            buffer_learner_step.append(learner_step)
            buffer_actor_step.append(actor_step)
            buffer_cur_size.append(cur_size)
    return np.array(buffer_actor_step), np.array(buffer_learner_step), np.array(buffer_cur_size)


# class Options:
#     def __init__(self, num_nodes, num_workers):
#         self.num_nodes = num_nodes
#         self.num_workers = num_workers

if __name__ == '__main__':

    # ray.init(local_mode=True)  # Local Mode
    ray.init()  #specify cluster address here
    node_ps = []
    node_buffer = []
    opt = Options(FLAGS.num_nodes, FLAGS.num_workers)
    opt.isOnline = 0
    # optref = ray.put(opt)
    for node_index in range(FLAGS.num_nodes):
        node_ps.append(
            ParameterServer.remote(opt, node_index,
                                        [f'{os.getcwd()}/{f}' for f in
                                            glob.glob('models/*') if
                                            "." not in f], ""))
        print(f"Node{node_index} Parameter Server all set.")

        node_buffer.append(
            {model_type: ReplayBuffer.remote(opt, node_index, model_type) for
             model_type in model_types})
        print(f"Node{node_index} Experience buffer all set.")

        #create buffer path
        buffer_save_path = opt.save_dir+f'/buffer/{str(node_index)}/'
        if not os.path.exists(buffer_save_path):
            os.mkdir(buffer_save_path)

        for i in range(FLAGS.num_workers):
            worker_rollout.remote(node_ps[node_index], node_buffer[node_index], opt)
                                                                                    
            time.sleep(0.19)
        print(f"Node{node_index} roll out worker all up.")

    print("Ray total resources:", ray.cluster_resources())
    print("available resources:", ray.available_resources())

    nodes_info = {
        "node_buffer": np.array(node_buffer),
        "num_nodes": opt.num_nodes
    }
    f_name = './nodes_info.pickle'
    with open(f_name, "wb") as pickle_out:
        pickle.dump(nodes_info, pickle_out)
        print("****** save nodes_info ******")

    task_train = worker_train.remote(node_ps[0], node_buffer, opt)

    task_test = worker_test.remote(node_ps[0], node_buffer, opt)
    ray.wait([task_test])
