import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

import scipy.signal

from RLQWOP.ActorCritic import Actor, Critic, ActorCritic
from gym.spaces import Box, Discrete


def mlp(sizes, activation, output_activation=nn.Sigmoid):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPActor(Actor):    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, observation):
        probs = self.logits_net(observation)
        return Bernoulli(probs=probs)

    def _log_prob_from_distribution(self, distribution, action):
        probas = distribution.log_prob(action)
        return torch.sum(probas, dim=1)

class MLPCritic(Critic):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(ActorCritic):
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]   

        self.pi = MLPActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)