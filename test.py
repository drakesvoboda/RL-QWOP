from Environment import Environment
from ppo.ppo import PPOBuffer

import cv2
import time

import numpy as np

from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

env = Environment()

while True:
    obs = env.obs()

    print(obs.shape)

    cv2.imshow("asdf", obs[0,0])
    cv2.waitKey(0)

"""
mpi_fork(4)

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

buf = PPOBuffer((1,4,4), 4, 2)

buf.store(np.zeros((1, 4, 4)), np.zeros(4), 0, 0, 0)
buf.store(np.eye(4)[np.newaxis,...], np.ones(4), 1, 1, 1)

buf.finish_path()

data = buf.mpi_gather()

if rank == 0:
    print(f"PROC {rank} has: ")
    print(data["obs"])
    print(data["act"])
    print(data["logp"])"""