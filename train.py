from collections import deque
import random

import torch
import numpy as np
import pandas as pd
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt

from ddpg import DDPGAgent


def ddpg(agent: DDPGAgent, env, brain_name, n_episodes=10000):
    scores_deque = deque(maxlen=100)
    scores_list = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = np.concatenate(env_info.vector_observations, axis=0)
        agent.reset()
        scores = np.zeros(2)
        while True:
            actions = agent.act(states, i_episode, add_noise=True)
            env_info = env.step(actions)[brain_name]
            next_states = np.concatenate(env_info.vector_observations, axis=0)  # get next state (for each agent)
            rewards = np.array([np.sum(env_info.rewards)])  # get reward (for each agent)
            scores += np.array(env_info.rewards)
            dones = np.any(env_info.local_done)  # see if episode finished

            agent.step(states, actions, rewards, next_states, dones)

            states = next_states
            if dones:
                break

        scores_deque.append(np.max(scores))
        scores_list.append(np.max(scores))

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if np.mean(scores_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_deque)))
            break
    return scores_list


def main():
    env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")
    brain_name = env.brain_names[0]

    env_info = env.reset(train_mode=True)[brain_name]

    action_size = 4

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1] * 2

    ddpg_agent = DDPGAgent(state_size=state_size, action_size=action_size, random_seed=42)
    scores = ddpg(ddpg_agent, env, brain_name, n_episodes=10000)


if __name__ == '__main__':
    main()