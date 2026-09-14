import sys
sys.path.insert(0, 'src')

import gymnasium as gym
import gym_qwop.envs  # Register environments

from rlqwop.CNNActorCritic import CNNActorCritic
from rlqwop.MLPActorCritic import MLPActorCritic
from rlqwop.PPO import ppo

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--resume', type=str, default=None,
                         help='Path to a model.pt checkpoint to resume training from')
    args = parser.parse_args()

    logger_kwargs = dict(output_dir=f'logs/ppo_qwop_{args.seed}')

    ppo(lambda : gym.make("qwop-v0", ticks_per_step=3, render_mode="human"),
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=(64, 64)),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        pi_lr=3e-4,
        vf_lr=1e-3,
        target_kl=0.02,
        train_pi_iters=10,
        train_v_iters=10,
        num_envs=args.cpu,
        resume_from=args.resume)