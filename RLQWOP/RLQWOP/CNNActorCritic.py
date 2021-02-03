import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

import scipy.signal

from RLQWOP.ActorCritic import Actor, Critic, ActorCritic

class CNNActor(Actor):
    def __init__(self, observations_space, action_space): 
        super().__init__()

        obs_shape = observations_space.shape
        act_shape = action_space.shape

        flat_dim = 16 * obs_shape[1]//8 * obs_shape[2]//8
        
        self.model = nn.Sequential(
            nn.Conv2d(act_shape[0], 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((2,2)),    
            nn.Flatten(),
            nn.Linear(flat_dim, act_shape[0]),
            nn.Sigmoid(),
        )
        
    def _distribution(self, observation):
        probs = self.model(observation)
        return Bernoulli(probs=probs)
    
    def _log_prob_from_distribution(self, distribution, action):
        probas = distribution.log_prob(action)
        return torch.sum(probas, dim=1)
    
class CNNCritic(Critic):
    def __init__(self, observations_space): 
        super().__init__()
        
        obs_shape = observations_space.shape
        act_shape = observations_space.shape

        flat_dim = 16 * obs_shape[1]//8 * obs_shape[2]//8
        
        self.model = nn.Sequential(
            nn.Conv2d(act_shape[0], 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((2,2)),    
            nn.Flatten(),
            nn.Linear(flat_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, observation):
        return torch.squeeze(self.model(observation), -1) # Critical to ensure v has right shape.

class CNNActorCritic(ActorCritic):
    def __init__(self, observations_space, action_space):
        super().__init__()

        self.pi = CNNActor(observations_space, action_space)
        
        # build value function
        self.v  = CNNCritic(observations_space)