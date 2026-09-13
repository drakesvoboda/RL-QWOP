import sys
sys.path.insert(0, 'rl-qwop/src')

import cv2
import gymnasium as gym
import time
import numpy as np

from rlqwop.CNNActorCritic import CNNActorCritic
from rlqwop.PPO import PPOBuffer
from rlqwop.spinup_utils import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

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