import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

import scipy.signal

class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class CNNActor(Actor):
    def __init__(self): 
        super(CNNActor, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
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
            nn.Linear(16 * 4 * 8, 4),
            nn.Sigmoid(),
        )
        
    def _distribution(self, observation):
        probs = self.model(observation)
        return Bernoulli(probs=probs)
    
    def _log_prob_from_distribution(self, distribution, action):
        probas = distribution.log_prob(action)
        return torch.sum(probas, dim=1)
    
class CNNCritic(nn.Module):
    def __init__(self): 
        super(CNNCritic, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
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
            nn.Linear(16 * 4 * 8, 1)
        )
        
    def forward(self, observation):
        return torch.squeeze(self.model(observation), -1) # Critical to ensure v has right shape.

class CNNActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.pi = CNNActor()
        
        # build value function
        self.v  = CNNCritic()

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]