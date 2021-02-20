import gym

from RLQWOP.CNNActorCritic import CNNActorCritic
from RLQWOP.MLPActorCritic import MLPActorCritic
from RLQWOP.PPO import ppo

from spinup.utils.mpi_tools import mpi_fork

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

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs('PPO', args.seed)

    ppo(lambda : gym.make("gym_qwop:qwop-v0"), actor_critic=MLPActorCritic, gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, pi_lr=1e-6, vf_lr=1e-4, target_kl=0.05, train_pi_iters=80, train_v_iters=80)