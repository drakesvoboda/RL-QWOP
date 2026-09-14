import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import scipy.signal

from rlqwop.ActorCritic import Actor, Critic, ActorCritic
from gymnasium.spaces import Box, Discrete


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # No output activation: Categorical(logits=...) applies softmax internally, so
        # the net should output raw logits, not squashed probabilities.
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, observation):
        logits = self.logits_net(observation)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, distribution, action):
        return distribution.log_prob(action)

class MLPCritic(Critic):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        # No output activation: V(s) is a real-valued return estimate, not a probability -
        # squashing it through the default Sigmoid would cap it to (0, 1), which can't
        # represent the actual reward scale (e.g. velx * abs(velx) alone can exceed 1).
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation, output_activation=nn.Identity)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(ActorCritic):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(128,), activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]

        print('obs_dim', obs_dim)

        # Discrete(16): the actor's output layer has one logit per action (one per
        # combination of the 4 keys), not action_space.shape[0] (Discrete spaces have no
        # shape in that sense - shape is () - the actual size is .n).
        act_dim = action_space.n if isinstance(action_space, Discrete) else action_space.shape[0]

        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation)
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)