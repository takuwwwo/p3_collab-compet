import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from model import ActorNetwork, CriticNetwork
from utils import soft_update, hard_update
from settings import *


device = torch.device("cpu")


class DDPGAgent():
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        random.seed(random_seed)

        self.actor_local = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, random_seed).to(device)
        hard_update(self.actor_target, self.actor_local)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = CriticNetwork(state_size, action_size, random_seed).to(device)
        self.critic_target = CriticNetwork(state_size, action_size, random_seed).to(device)
        hard_update(self.critic_target, self.critic_local)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=0.)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.t_step = 0
        self.noise = RandomNoise(self.action_size,
                                 NOISE_START, NOISE_END, NOISE_DECAY,
                                 BEGIN_TRAINING_AT, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(states, actions, rewards, next_states, dones)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for _ in range(NUM_UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

                soft_update(self.critic_target, self.critic_local, TAU)
                soft_update(self.actor_target, self.actor_local, TAU)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        actions_target = self.actor_target(next_states).squeeze(1)
        actions_pred = self.actor_local(states).squeeze(1)

        # rewards = rewards.unsqueeze(-1)
        # dones = dones.unsqueeze(-1)
        # print(rewards.shape, dones.shape)

        q_targets_next = self.critic_target(next_states.reshape(next_states.shape[0], -1),
                                            actions_target.reshape(next_states.shape[0], -1)).squeeze(1)

        q_targets = rewards.squeeze(1) + (gamma * q_targets_next * (1 - dones))

        q_expected = self.critic_local(states.reshape(states.shape[0], -1), actions.reshape(actions.shape[0], -1)).squeeze(1)

        critic_loss = F.mse_loss(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic_local(states.reshape(states.shape[0], -1),
                                        actions_pred.reshape(actions_pred.shape[0], -1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def act(self, states, i_episode=0, add_noise=True):
        state = torch.from_numpy(states.reshape(1, -1)).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample(i_episode)
        actions = np.clip(action, -1, 1)
        return actions[0]

    def reset(self):
        self.noise.reset()


class RandomNoise:
    """Random noise process."""
    def __init__(self, size, weight, min_weight, noise_decay,
                 begin_noise_at, seed):
        self.size = size
        self.weight_start = weight
        self.weight = weight
        self.min_weight = min_weight
        self.noise_decay = noise_decay
        self.begin_noise_at = begin_noise_at
        self.seed = random.seed(seed)

    def reset(self):
        self.weight = self.weight_start

    def sample(self, i_episode):
        pwr = max(0, i_episode - self.begin_noise_at)
        if pwr > 0:
            self.weight = max(self.min_weight, self.noise_decay**pwr)
        return self.weight * 0.5 * np.random.standard_normal(self.size)