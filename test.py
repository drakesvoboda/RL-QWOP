from RLQWOP.QWOPEnvironment import QWOPEnvironment
from RLQWOP.CNNActorCritic import CNNActorCritic
from RLQWOP.PPO import PPOBuffer
from RLQWOP.RandomAgent import RandomAgent

import cv2
import gym
import time

import numpy as np

from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from stable_baselines.common.env_checker import check_env

# env = Environment()

env = gym.make('gym_qwop:qwop-v0')

# check_env(env)

total = 0

obs = env.reset()

while True:
    start = time.time()
    env.throttle()
    state = env.get_state()
    if (state["terminal"]): env.reset()

    obs = env.obs(state)
    rew = env.compute_reward(state)
    end = time.time()
    print(rew)
    #cv2.imshow("asdf", obs)
    #cv2.waitKey(0)
    # rew = env.get_state()[0]
    #total += rew

    # print(rew)

    time.sleep(1/14)