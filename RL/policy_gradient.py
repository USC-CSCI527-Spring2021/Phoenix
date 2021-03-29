import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam


class PolicyGradientNN(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(PolicyGradientNN, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation="relu")
        self.fc2 = Dense(self.fc2_dims, activation="relu")
        self.pi = Dense(n_actions, activation="softmax")

    def call(self, state):
        """

        """
        first_layer_value = self.fc1(state)
        second_layer_value = self.fc2(first_layer_value)
        final_value = self.pi(second_layer_value)
        return final_value


class Agent:
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=4, layer1_size=256, layer2_size=256):
        self.gamma = gamma
        self.lr = alpha
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy = PolicyGradientNN(n_actions, layer1_size, layer2_size)
        self.policy.compile(optimizer=Adam(learning_rate=self.lr))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()
        return action.numpy()[0]

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = np.array(self.reward_memory)  ## Need to be changed
        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum = G_sum + rewards[k] * discount
                discount = discount * self.gamma
            G[t] = G_sum
        with tf.GradientTape() as U:
            loss = 0
            for index, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[index])
                loss += -g * tf.squeeze(log_prob)
        grad = U.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(grad, self.policy.trainable_variables))
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

