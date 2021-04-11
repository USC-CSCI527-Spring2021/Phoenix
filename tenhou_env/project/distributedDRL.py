import ray
import numpy as np
import os
import sys
import multiprocessing
import json
import copy
from actor_learner import Learner, Actor
import pickle
import tensorflow as tf
import time
from options import Options

flags = tf.compat.v1.flags
FLAGS = tf.compat.v1.flags.FLAGS
model_types = ['chi', 'pon', 'kan', 'riichi', 'discard']
flags.DEFINE_integer("num_nodes", 1, "number of nodes")
flags.DEFINE_integer("num_workers", 12, "number of workers")



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
        
    def store(self, obs, rew, pred, act):
        # self.obs1_buf[self.ptr] = obs
        # self.obs2_buf[self.ptr] = next_obs
        # self.acts_buf[self.ptr] = act
        # self.rews_buf[self.ptr] = rew
        self.buf[self.ptr] = [obs, rew, pred, act]
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
        self.buf, self.ptr, self.size, self.max_size, self.learner_steps, self.actor_steps = info['buffer'], \
            info['ptr'], info['size'], info['max_size'], info['learner_steps'], info['actor_steps']
        print("****** buffer " + self.buffer_type + " restored! ******")
        print("****** buffer " + self.buffer_type + " infos:", self.ptr, self.size, self.max_size,
              self.actor_steps, self.learner_steps)



class Cache():

    def __init__(self, node_buffer):
        # cache for training data and model weights
        print('os.pid:', os.getpid())
        self.node_buffer = node_buffer
        self.q1 = {k: multiprocessing.Queue(12) for k in model_types}
        self.q2 = multiprocessing.Queue(5)
        self.p1 = multiprocessing.Process(target=self.ps_update, args=(self.q1, self.q2, self.node_buffer))
        self.p1.daemon = True

    def ps_update(self, q1, q2, node_buffer):
        print('os.pid of put_data():', os.getpid())

        node_idx = np.random.choice(opt.num_nodes, 1)[0]
        for model_type in model_types:
            q1[model_type].put(copy.deepcopy(ray.get(node_buffer[node_idx][model_type].sample_batch.remote())))

        while True:
            for model_type in model_types:

                if q1[model_type].qsize() < 10:
                    node_idx = np.random.choice(opt.num_nodes, 1)[0]
                    q1[model_type].put(copy.deepcopy(ray.get(node_buffer[node_idx][model_type].sample_batch.remote())))

            if not q2.empty():
                keys, values = q2.get()
                [node_ps[i].push.remote(keys, values) for i in range(opt.num_nodes)]

    def start(self):
        self.p1.start()
        self.p1.join(10)

    def end(self):
        self.p1.terminate()

@ray.remote
class ParameterServer:
    def __init__(self, opt, weights_file, checkpoint_path, ps_index):
        # each node will have a Parameter Server

        self.opt = opt
        self.learner_step = 0


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

        self.weights = None

        # if not checkpoint_path:
        #     checkpoint_path = opt.save_dir + "/checkpoint"

        # if opt.recover:
        #     with open(checkpoint_path + "/checkpoint_weights.pickle", "rb") as pickle_in:
        #         self.weights = pickle.load(pickle_in)
        #         print("****** weights restored! ******")

        # if weights_file:
        #     try:
        #         with open(weights_file, "rb") as pickle_in:
        #             self.weights = pickle.load(pickle_in)
        #             print("****** weights restored! ******")
        #     except:
        #         print("------------------------------------------------")
        #         print(weights_file)
        #         print("------ error: weights file doesn't exist! ------")
        #         exit()

        # if not opt.recover and not weights_file:
        #     values = [value.copy() for value in values]
        #     self.weights = dict(zip(keys, values))

    def push(self, model_type, weights):
        self.weights[model_type] = weights
        self.learner_step += self.opt.push_freq

    def pull(self, keys):
        return self.weights

    def get_weights(self):
        return copy.deepcopy(self.weights)

    # save weights to disk
    def save_weights(self):
        with open(self.opt.save_dir + "/checkpoint/" + "checkpoint_weights.pickle", "wb") as pickle_out:
            pickle.dump(self.weights, pickle_out)

@ray.remote(num_cpus=1, num_gpus=1, max_calls=1)    #centralized training
def worker_train(ps, node_buffer, opt, model_type):
    agent = Learner(opt, model_type)
    weights = ray.get(ps.pull.remote(model_type))
    agent.set_weights(weights)

    cache = Cache(node_buffer)
    cache.start()

    cnt = 1
    while True:
        batch = cache.q1[model_type].start()
        agent.train(batch, cnt)

        if cnt % opt.push_freq == 0:
            cache.q2.put(agent.get_weights)
        cnt += 1

@ray.remote
def worker_rollout(ps, replay_buffer, opt):
    agent = Actor(opt, job='worker', buffer=replay_buffer)
    
    while True:
        weights = ray.get(ps.pull.remote())
        agent.set_weights(weights)
        agent.run()

@ray.remote
def worker_test(ps, node_buffer, opt):
    agent = Actor(opt, job="test", buffer=ReplayBuffer)
    init_time = time.time()
    save_times = 0
    checkpoint_times = 0

    while True:
        weights = ray.get(ps.get_weights.remote())
        agent.set_weights(weights)
        last_actor_step, last_learner_step, _ = get_al_status(node_buffer)
        start_time = time.time()

        for i in range(10):
            agent.run()

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
            buffer_save_op = [node_buffer[node_index][model_type].save.remote() for model_type in model_types) for node_index in range(opt.num_nodes)]
            ray.wait(buffer_save_op + ps_save_op, num_returns=opt.num_nodes * 6)       #5 models + ps

            print("total time for saving :", time.time() - save_start_time)
            checkpoint_times = total_time // opt.checkpoint_freq


def get_al_status(node_buffer):

    buffer_learner_step = []
    buffer_actor_step = []
    buffer_cur_size = []

    for node_index in range(opt.num_nodes):
        for model_type in range(model_types):
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
    ray.init()  #specify cluster address here

    node_ps = []
    node_buffer = []
    opt = Options(FLAGS.num_nodes, FLAGS.num_workers)

    for node_index in range(FLAGS.num_nodes):
        node_ps.append(ParameterServer.options(resources={"node"+str(node_index):1}).remote(opt, node_index))
        print(f"Node{node_index} Parameter Server all set.")


        node_buffer.append([ReplayBuffer.options(resources={"node"+str(node_index):1}).remote(opt, node_index, model_type) for model_type in model_types])
        print(f"Node{node_index} Experience buffer all set.")

        for i in range(FLAGS.num_workers):
            worker_rollout.options(resources={"node"+str(node_index):1}).remote(node_ps[node_index], node_buffer[node_index], opt)
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

    task_train = []
    for model_type in model_types:
        task_train.append(worker_train.options(resources={"node0": 1}).remote(node_ps[0], node_buffer, opt, model_type))

