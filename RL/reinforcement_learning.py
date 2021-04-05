## Main driver for the reinforcement learning...
import numpy as np
from numpy.lib.histograms import histogram
from policy_gradient import Agent
from environment import LearningEnv


if __name__ == "main":
    agent = Agent(0.0005, 0.99, 4, 256, 256)

    env = LearningEnv()
    score_history = []
    num_episodes = 2000

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score = score + reward
        score_history.append(score)
        agent.learn()
        avg_score = np.mean(score_history[-100, :])
        print("episode: ", i, "score: %.1f" % score, "average score %.1f" % avg_score)


