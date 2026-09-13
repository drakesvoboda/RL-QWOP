import sys
sys.path.insert(0, 'rl-qwop/src')

import gymnasium as gym
import gym_qwop.envs  # Register environments

from rlqwop.CNNActorCritic import CNNActorCritic
from rlqwop.MLPActorCritic import MLPActorCritic
from rlqwop.PPO import ppo
from rlqwop.spinup_utils import mpi_fork

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    logger_kwargs = dict(output_dir=f'logs/ppo_qwop_{args.seed}')

    ppo(lambda : gym.make("qwop-websocket-v0"),  # Use "qwop-websocket-v0" for 10-20x speedup (requires game server WebSocket support) 
        actor_critic=MLPActorCritic, 
        gamma=args.gamma,
        seed=args.seed, 
        steps_per_epoch=args.steps, 
        epochs=args.epochs,
        logger_kwargs=logger_kwargs, 
        pi_lr=1e-6, 
        vf_lr=1e-4, 
        target_kl=0.05, 
        train_pi_iters=80, 
        train_v_iters=80)