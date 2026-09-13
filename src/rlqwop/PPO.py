""" Custom implementation of proximal policy optimization. Multiple parallel actors simulate the environment and 
run local copies of the policy. Experiences are gathered by a learner processes that performs SGD to update the policy.
The new policy is then sent back to the actor processes. 
"""

import time
from functools import reduce
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, RandomSampler

try:
    from mpi4py import MPI
    _HAS_MPI = True
except ImportError:
    _HAS_MPI = False
    class _MockMPI:
        class COMM_WORLD:
            @staticmethod
            def Get_size():
                return 1
            @staticmethod
            def Get_rank():
                return 0
            @staticmethod
            def Gatherv(*args, **kwargs):
                pass
        class FLOAT:
            pass
    MPI = _MockMPI()

import gymnasium as gym

from rlqwop.MLPActorCritic import MLPActorCritic
from rlqwop.spinup_utils import (
    EpochLogger,
    combined_shape, discount_cumsum, count_vars,
    setup_pytorch_for_mpi, sync_params, mpi_avg_grads,
    mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
)

# Create a mock core module for compatibility
class _CoreModule:
    combined_shape = staticmethod(combined_shape)
    discount_cumsum = staticmethod(discount_cumsum)
    count_vars = staticmethod(count_vars)

core = _CoreModule()


class BufferDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, idx):
        return self.data['obs'][idx], self.data['act'][idx], self.data['ret'][idx], self.data['adv'][idx], self.data['logp'][idx]


def actor_loss(pi, adv, logp, logp_old, clip_ratio): 
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info

def critic_loss(pred, ret):  
    return ((pred - ret)**2).mean()

def run_update(ac, data, pi_optimizer, vf_optimizer, num_iters, clip_ratio, target_kl, logger, batch_size=None):
    dataset = BufferDataset(data)

    # Adaptive batch size: 4-8 batches per epoch max
    if batch_size is None:
        batch_size = max(4, len(dataset) // 8)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ac.v = ac.v.to(device)
    ac.pi = ac.pi.to(device)

    kl_sum = 0
    kl_total = 0
    loss_pi_sum = 0.0
    loss_v_sum = 0.0
    approx_kl_sum = 0.0
    ent_sum = 0.0
    clipfrac_sum = 0.0
    n_updates = 0

    # Joint policy and value optimization
    for i in trange(num_iters):
        for obs, act, ret, adv, logp_old in dataloader:
            obs = obs.to(device)
            act = act.to(device)
            ret = ret.to(device)
            adv = adv.to(device)
            logp_old = logp_old.to(device)

            # Policy update
            pi, logp = ac.pi(obs, act)
            pi_loss, pi_info = actor_loss(pi, adv, logp, logp_old, clip_ratio)

            # Value update
            pred_return = ac.v(obs)
            vf_loss = critic_loss(pred_return, ret)

            # Combined loss
            loss = pi_loss + vf_loss

            pi_optimizer.zero_grad()
            vf_optimizer.zero_grad()
            loss.backward()
            pi_optimizer.step()
            vf_optimizer.step()

            # Collect statistics
            loss_pi_sum += pi_loss.item()
            loss_v_sum += vf_loss.item()
            approx_kl_sum += pi_info['kl']
            ent_sum += pi_info['ent']
            clipfrac_sum += pi_info['cf']
            n_updates += 1

            kl_sum += (logp_old - logp).sum().item()
            kl_total += logp_old.shape[0]

        # Early stopping per epoch of policy updates
        if kl_total > 0:
            avg_kl = kl_sum / kl_total
            if avg_kl > 1.5 * target_kl:
                logger.log(f'Early stopping at epoch {i} due to reaching max kl: {avg_kl:.4f}')
                break

    ac.v = ac.v.cpu()
    ac.pi = ac.pi.cpu()

    # Log actual statistics
    if n_updates > 0:
        logger.store(
            LossPi=loss_pi_sum / n_updates,
            LossV=loss_v_sum / n_updates,
            KL=approx_kl_sum / n_updates,
            Entropy=ent_sum / n_updates,
            ClipFrac=clipfrac_sum / n_updates,
            DeltaLossPi=0,
            DeltaLossV=0
        )

    return ac.cpu()


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def normalize_advantages(self):
        """Normalize advantages with safe handling of zero variance."""
        adv_mean = self.adv_buf.mean()
        adv_std = self.adv_buf.std()
        if adv_std > 1e-8:
            self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        else:
            self.adv_buf = self.adv_buf - adv_mean

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def mpi_gather(self, root=0):
        assert self.ptr == self.max_size    # buffer has to be full before you can get

        self.ptr, self.path_start_idx = 0, 0

        comm = MPI.COMM_WORLD

        num_procs = comm.Get_size()
        rank = comm.Get_rank()
        
        size = self.max_size * num_procs
        offsets = np.arange(0, size, self.max_size)

        flat_obs = reduce(lambda a,b: a*b, self.obs_dim)
        
        obs_buf  = np.zeros(size * flat_obs, dtype=np.float32)
        act_buf  = np.zeros(size * self.act_dim, dtype=np.float32)
        adv_buf  = np.zeros(size, dtype=np.float32)
        rew_buf  = np.zeros(size, dtype=np.float32)
        ret_buf  = np.zeros(size, dtype=np.float32)
        val_buf  = np.zeros(size, dtype=np.float32)
        logp_buf = np.zeros(size, dtype=np.float32)

        comm.Gatherv(self.obs_buf.flatten(), [obs_buf, size * flat_obs, offsets*flat_obs, MPI.FLOAT], root=root)
        comm.Gatherv(self.act_buf.flatten(), [act_buf, size * self.act_dim, offsets*self.act_dim, MPI.FLOAT], root=root)
        comm.Gatherv(self.adv_buf, [adv_buf, size, offsets, MPI.FLOAT], root=root)
        comm.Gatherv(self.rew_buf, [rew_buf, size, offsets, MPI.FLOAT], root=root)
        comm.Gatherv(self.ret_buf, [ret_buf, size, offsets, MPI.FLOAT], root=root)
        comm.Gatherv(self.val_buf, [val_buf, size, offsets, MPI.FLOAT], root=root)
        comm.Gatherv(self.logp_buf, [logp_buf, size, offsets, MPI.FLOAT], root=root)

        if rank != root: return None

        obs_buf = obs_buf.reshape(core.combined_shape(size, self.obs_dim))
        act_buf = act_buf.reshape(core.combined_shape(size, self.act_dim))

        # Safe advantage normalization
        adv_mean = adv_buf.mean()
        adv_std = adv_buf.std()
        if adv_std > 1e-8:
            adv_buf = (adv_buf - adv_mean) / (adv_std + 1e-8)
        else:
            adv_buf = adv_buf - adv_mean

        data = dict(obs=obs_buf, act=act_buf, ret=ret_buf, adv=adv_buf, logp=logp_buf)

        return data


def ppo(env_fn, actor_critic=MLPActorCritic, ac_kwargs={}, seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # Prepare for interaction with environment
    start_time = time.time()
    reset_result = env.reset()
    o = reset_result if isinstance(reset_result, np.ndarray) else reset_result[0]
    ep_ret, ep_len = 0, 0
    ep_max_posx, ep_sum_posx = 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        with torch.no_grad():
            for t in range(local_steps_per_epoch):
                a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))

                a = a.squeeze()
                v = float(v.squeeze())
                logp = float(logp.squeeze())

                next_o, r, terminated, truncated, info = env.step(a)
                d = terminated or truncated
                ep_ret += r
                ep_len += 1

                # Track position for logging
                state = info.get('state', {})
                posx = state.get('posx', 0)
                ep_max_posx = max(ep_max_posx, posx)
                ep_sum_posx += posx

                # save and log
                buf.store(o, a, r, v, logp)
                logger.store(VVals=v)

                # Render to see agent progress
                if hasattr(env, 'render'):
                    env.render()

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == local_steps_per_epoch-1

                time.sleep(0.05)

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' %
                            ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
                        v = float(v.squeeze())
                    else:
                        v = 0
                    buf.finish_path(v)
                    # Log episode stats (both terminal and epoch-ended episodes)
                    if ep_len > 0:
                        logger.store(EpRet=ep_ret, EpLen=ep_len)
                        logger.store(EpMaxPosx=ep_max_posx, EpAvgPosx=ep_sum_posx/ep_len)
                    reset_result = env.reset()
                    o = reset_result if isinstance(reset_result, np.ndarray) else reset_result[0]
                    ep_ret, ep_len = 0, 0
                    ep_max_posx, ep_sum_posx = 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        data = buf.mpi_gather()

        if rank == 0: #Root process performs ppo update
            ac = run_update(ac, data, pi_optimizer, vf_optimizer, train_v_iters, clip_ratio, target_kl, logger, 1024)
            param = next(iter(ac.v.parameters()))
            
        sync_params(ac)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('EpMaxPosx', with_min_and_max=True)
        logger.log_tabular('EpAvgPosx', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        # logger.log_tabular('LossPi', average_only=True)
        # logger.log_tabular('LossV', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        # logger.log_tabular('DeltaLossV', average_only=True)
        # logger.log_tabular('Entropy', average_only=True)
        # logger.log_tabular('KL', average_only=True)
        # logger.log_tabular('ClipFrac', average_only=True)
        # logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()