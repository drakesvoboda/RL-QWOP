""" Custom implementation of proximal policy optimization.

Parallelism is vectorized, not multi-process-with-synced-policies: `num_envs` copies of the
environment run in subprocesses (gymnasium.vector.AsyncVectorEnv), stepped in lockstep from
this one process, which holds the only copy of the policy. Each iteration does one batched
forward pass across all `num_envs` observations rather than `num_envs` separate ones. This
fits the workload here - the environment (a real browser) is what's slow, not the policy
(a couple thousand params) - so parallelizing environment stepping is what matters; there's
no need for multiple policy copies, gradient synchronization, or cross-process communication
at all, since the single policy never leaves this process.
"""

import time
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, RandomSampler
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.vector.vector_env import AutoresetMode
from gymnasium.wrappers import TimeLimit

from rlqwop.MLPActorCritic import MLPActorCritic
from rlqwop.spinup_utils import EpochLogger, combined_shape, discount_cumsum, count_vars, RunningMeanStd

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


def actor_loss(pi, adv, logp, logp_old, clip_ratio, ent_coef=0.0):
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
    entropy = pi.entropy().mean()
    # Entropy bonus: without it, PPO has no pressure to keep trying varied actions once it
    # finds anything that survives - it's free to collapse onto the first workable gait
    # (e.g. a safe scoot) even if better ones exist, since nothing in the objective values
    # the exploration needed to find them.
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() - ent_coef * entropy

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=entropy.item(), cf=clipfrac)

    return loss_pi, pi_info

def critic_loss(pred, ret):  
    return ((pred - ret)**2).mean()

def run_update(ac, data, pi_optimizer, vf_optimizer, train_pi_iters, train_v_iters, clip_ratio,
               target_kl, logger, batch_size=None, ent_coef=0.0, max_grad_norm=0.5):
    dataset = BufferDataset(data)

    # Adaptive batch size: 4-8 batches per epoch max
    if batch_size is None:
        batch_size = max(4, len(dataset) // 8)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ac.v = ac.v.to(device)
    ac.pi = ac.pi.to(device)

    # --- Policy update: up to train_pi_iters passes over the buffer, stopping early if
    # this pass's average KL from the old policy exceeds the target. KL is computed fresh
    # per pass (not accumulated across passes) - averaging it in with earlier, typically
    # much smaller KL values would dilute the signal and let the policy drift far more per
    # update than target_kl is meant to allow before this triggers. ---
    loss_pi_sum = 0.0
    approx_kl_sum = 0.0
    ent_sum = 0.0
    clipfrac_sum = 0.0
    n_pi_updates = 0
    pi_iters_run = 0

    for i in trange(train_pi_iters, desc='pi'):
        pi_iters_run = i + 1
        kl_sum = 0.0
        kl_total = 0
        for obs, act, ret, adv, logp_old in dataloader:
            obs = obs.to(device)
            act = act.to(device)
            adv = adv.to(device)
            logp_old = logp_old.to(device)

            pi, logp = ac.pi(obs, act)
            pi_loss, pi_info = actor_loss(pi, adv, logp, logp_old, clip_ratio, ent_coef)

            pi_optimizer.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.pi.parameters(), max_grad_norm)
            pi_optimizer.step()

            loss_pi_sum += pi_loss.item()
            approx_kl_sum += pi_info['kl']
            ent_sum += pi_info['ent']
            clipfrac_sum += pi_info['cf']
            n_pi_updates += 1

            kl_sum += (logp_old - logp).sum().item()
            kl_total += logp_old.shape[0]

        avg_kl = kl_sum / kl_total if kl_total > 0 else 0.0
        if avg_kl > 1.5 * target_kl:
            logger.log(f'Early stopping policy update at iter {i} due to reaching max kl: {avg_kl:.4f}')
            break

    # --- Value function update: a fixed train_v_iters passes, independent of the policy's
    # early stopping above - value regression doesn't have the same on/off-policy drift
    # risk that motivates stopping the policy update early. ---
    loss_v_sum = 0.0
    n_v_updates = 0

    for _ in trange(train_v_iters, desc='v'):
        for obs, act, ret, adv, logp_old in dataloader:
            obs = obs.to(device)
            ret = ret.to(device)

            pred_return = ac.v(obs)
            vf_loss = critic_loss(pred_return, ret)

            vf_optimizer.zero_grad()
            vf_loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.v.parameters(), max_grad_norm)
            vf_optimizer.step()

            loss_v_sum += vf_loss.item()
            n_v_updates += 1

    ac.v = ac.v.cpu()
    ac.pi = ac.pi.cpu()

    # Log actual statistics
    if n_pi_updates > 0:
        logger.store(
            LossPi=loss_pi_sum / n_pi_updates,
            KL=approx_kl_sum / n_pi_updates,
            Entropy=ent_sum / n_pi_updates,
            ClipFrac=clipfrac_sum / n_pi_updates,
            StopIter=pi_iters_run,
        )
    if n_v_updates > 0:
        logger.store(LossV=loss_v_sum / n_v_updates)

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
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8) if adv_std > 1e-8 else self.adv_buf - adv_mean
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def combine_buffers(bufs):
    """Concatenate several envs' full local buffers (one per vectorized env) into a single
    training batch, normalizing advantages once across the combined pool - the vectorized
    equivalent of what the old MPI-style mpi_gather did across processes, except everything
    is already in this one process, so it's just concatenation, no communication."""
    for buf in bufs:
        assert buf.ptr == buf.max_size    # every buffer must be full before combining
        buf.ptr, buf.path_start_idx = 0, 0

    obs = np.concatenate([b.obs_buf for b in bufs], axis=0)
    act = np.concatenate([b.act_buf for b in bufs], axis=0)
    adv = np.concatenate([b.adv_buf for b in bufs], axis=0)
    ret = np.concatenate([b.ret_buf for b in bufs], axis=0)
    logp = np.concatenate([b.logp_buf for b in bufs], axis=0)

    adv_mean, adv_std = adv.mean(), adv.std()
    adv = (adv - adv_mean) / (adv_std + 1e-8) if adv_std > 1e-8 else adv - adv_mean

    return dict(obs=obs, act=act, ret=ret, adv=adv, logp=logp)


def _info_get(info_dict, key, i):
    """Read info_dict[key][i], gated by the parallel info_dict["_"+key][i] boolean mask -
    gymnasium's vector-info convention for keys not every env provides every step."""
    values = info_dict.get(key)
    mask = info_dict.get(f"_{key}")
    if values is None or mask is None or not mask[i]:
        return None
    return values[i]


def _vector_scalar(infos, key, i, done):
    """Extract env i's `key` (a top-level scalar info field, e.g. "posx"/"velx") for this
    step. On a step where env i auto-resets (SAME_STEP autoreset mode), the real
    terminal-step info nests one level deeper - under `final_info` - instead of at the top
    level. `final_info` itself isn't a per-env array like other info keys (its "value" is
    a whole nested vector-info dict shared across envs, keyed by field name), so it needs
    its own presence check via `_final_info` rather than going through `_info_get`."""
    if done:
        final_mask = infos.get("_final_info")
        if final_mask is None or not final_mask[i]:
            return 0
        return _info_get(infos.get("final_info", {}), key, i) or 0
    return _info_get(infos, key, i) or 0


def _vector_final_obs(infos, i):
    """Extract env i's actual terminal observation, before its SAME_STEP auto-reset."""
    return _info_get(infos, "final_obs", i)


def ppo(env_fn, actor_critic=MLPActorCritic, ac_kwargs={}, seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, num_envs=1, ent_coef=0.01,
        max_grad_norm=0.5, normalize_obs=True, resume_from=None):
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

        num_envs (int): Number of environments to run in parallel (each in its own
            subprocess/browser tab), stepped in lockstep from this one process. There's
            only ever one policy - no per-process copies or weight synchronization -
            since every env's experience lands in this same process either way.

        ent_coef (float): Entropy bonus coefficient, subtracted from the policy loss
            (loss_pi -= ent_coef * entropy). Without this, PPO has no explicit pressure to
            keep trying varied actions once it finds anything that survives, so it can
            collapse onto a safe-but-mediocre gait rather than exploring toward a better
            one. Higher values push harder for continued exploration at the cost of
            slower convergence to a specific behavior.

        max_grad_norm (float): Max gradient norm for the policy and value networks
            (clipped separately), so one unusually large batch can't cause a single
            destructively large update.

        normalize_obs (bool): Maintain a running mean/std over observations and feed the
            network normalized obs instead of raw ones (see RunningMeanStd) - raw obs mix
            very different natural scales (torso height, joint angles, velocities), which
            a freshly initialized network has no principled way to weigh comparably.

        resume_from (str): Path to a model.pt checkpoint (from a previous
            EpochLogger.save_state()) to resume from - loads the actor-critic's weights
            and, if normalize_obs is True and the checkpoint has one, the observation
            normalizer's running stats, before training starts. ac_kwargs must produce
            the same architecture the checkpoint was saved from (load_state_dict fails
            on a shape mismatch). Optimizer state isn't saved/restored, so Adam's moment
            estimates restart from scratch - that typically settles within a handful of
            updates and isn't worth the extra checkpoint complexity. None (default)
            starts training from a fresh, randomly initialized policy.

    """

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    # num_envs copies of the environment run in parallel subprocesses (one browser tab
    # each), all pointing at the same shared static-asset game server. TimeLimit enforces
    # max_ep_len as a standard gymnasium truncation, so the vector env's own autoreset
    # machinery handles it uniformly instead of needing custom bookkeeping in this loop.
    def make_env():
        return TimeLimit(env_fn(), max_episode_steps=max_ep_len)

    if num_envs > 1:
        envs = AsyncVectorEnv([make_env] * num_envs, context="fork", autoreset_mode=AutoresetMode.SAME_STEP)
    else:
        envs = SyncVectorEnv([make_env], autoreset_mode=AutoresetMode.SAME_STEP)

    try:
        obs_dim = envs.single_observation_space.shape
        # .shape rather than .shape[0]: works generically for both Box/MultiBinary action
        # spaces ((n,)) and Discrete ((), a scalar action index per timestep - the buffer
        # ends up storing plain ints rather than per-key vectors).
        act_dim = envs.single_action_space.shape

        # Create the actor-critic module - a single instance. There's only ever one
        # policy: no per-process copies, no weight broadcasting, since every env's
        # experience already lands in this one process.
        ac = actor_critic(envs.single_observation_space, envs.single_action_space, **ac_kwargs)

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
        logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # One experience buffer per env, each holding that env's local share of the
        # epoch's total step budget.
        local_steps_per_epoch = int(steps_per_epoch / num_envs)
        bufs = [PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam) for _ in range(num_envs)]

        # Raw obs mix very different natural scales (torso height, joint angles,
        # velocities) - see RunningMeanStd. update_and_normalize is for the per-step
        # batches (updates the running stats with this batch, then normalizes it);
        # normalize_only is for one-off single observations pulled from info['final_obs']
        # (the real terminal obs on a step where an env auto-reset), which shouldn't also
        # be folded into the running stats a second time alongside the batch it came from.
        obs_rms = RunningMeanStd(shape=obs_dim) if normalize_obs else None

        def update_and_normalize(x):
            if obs_rms is None:
                return np.asarray(x, dtype=np.float32)
            obs_rms.update(x)
            return obs_rms.normalize(x).astype(np.float32)

        def normalize_only(x):
            if obs_rms is None:
                return np.asarray(x, dtype=np.float32)
            return obs_rms.normalize(x).astype(np.float32)

        if resume_from is not None:
            checkpoint = torch.load(resume_from, map_location='cpu', weights_only=False)
            ac.load_state_dict(checkpoint['model_state_dict'])
            if obs_rms is not None and 'obs_rms' in checkpoint:
                obs_rms.mean = checkpoint['obs_rms']['mean']
                obs_rms.var = checkpoint['obs_rms']['var']
                obs_rms.count = checkpoint['obs_rms']['count']
            logger.log(f'Resumed from checkpoint: {resume_from}')

        # Set up optimizers for policy and value function
        pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
        vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

        # Set up model saving
        logger.setup_pytorch_saver(ac)

        # Prepare for interaction with environment
        start_time = time.time()
        obs, _ = envs.reset(seed=seed)
        obs = update_and_normalize(obs)
        ep_ret = np.zeros(num_envs)
        ep_len = np.zeros(num_envs, dtype=int)
        ep_max_posx = np.zeros(num_envs)
        ep_sum_posx = np.zeros(num_envs)
        ep_sum_velx = np.zeros(num_envs)
        current_posx = np.zeros(num_envs)

        # Main loop: collect experience in the envs and update/log each epoch
        for epoch in range(epochs):
            with torch.no_grad():
                for t in range(local_steps_per_epoch):
                    a, v, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))

                    next_obs, rewards, terminateds, truncateds, infos = envs.step(a)
                    next_obs = update_and_normalize(next_obs)
                    epoch_ended = t == local_steps_per_epoch - 1

                    for i in range(num_envs):
                        done = bool(terminateds[i]) or bool(truncateds[i])

                        ep_ret[i] += rewards[i]
                        ep_len[i] += 1

                        posx = _vector_scalar(infos, "posx", i, done)
                        ep_max_posx[i] = max(ep_max_posx[i], posx)
                        ep_sum_posx[i] += posx
                        ep_sum_velx[i] += _vector_scalar(infos, "velx", i, done)
                        # Cumulative within the episode already (see QWOPEnv.step) - just
                        # read the current value, don't sum/max it across steps.
                        stride_count = _vector_scalar(infos, "stride_count", i, done)
                        current_posx[i] = 0.0 if done else posx

                        bufs[i].store(obs[i], a[i], rewards[i], v[i], logp[i])
                        logger.store(VVals=v[i])

                        if done or epoch_ended:
                            if epoch_ended and not done:
                                print('Warning: trajectory cut off by epoch at %d steps.' %
                                    ep_len[i], flush=True)
                            # Bootstrap the value target unless the episode reached a true
                            # terminal state (fell over) - nothing beyond that to bootstrap.
                            if done and terminateds[i]:
                                v_boot = 0.0
                            else:
                                if done:
                                    # Real terminal obs from info['final_obs'] hasn't been
                                    # through update_and_normalize yet (next_obs[i] here is
                                    # the post-autoreset obs, not this) - normalize it the
                                    # same way, just without double-counting it into the
                                    # running stats already updated for this batch.
                                    last_obs = normalize_only(_vector_final_obs(infos, i))
                                else:
                                    last_obs = next_obs[i]
                                _, v_boot, _ = ac.step(torch.as_tensor(last_obs, dtype=torch.float32).unsqueeze(0))
                                v_boot = float(v_boot.squeeze())
                            bufs[i].finish_path(v_boot)
                            if ep_len[i] > 0:
                                logger.store(EpRet=ep_ret[i], EpLen=ep_len[i])
                                logger.store(EpMaxPosx=ep_max_posx[i], EpAvgPosx=ep_sum_posx[i] / ep_len[i])
                                logger.store(EpAvgVelx=ep_sum_velx[i] / ep_len[i])
                                logger.store(EpStrideCount=stride_count)
                            ep_ret[i], ep_len[i] = 0, 0
                            ep_max_posx[i], ep_sum_posx[i], ep_sum_velx[i] = 0, 0, 0

                    # Live progress line: current position per env, overwritten in place
                    # (\r, no newline) so it doesn't spam the terminal. Uses the posx
                    # already extracted above rather than calling .render() on the actual
                    # env objects, which only works for num_envs=1 (SyncVectorEnv) -
                    # AsyncVectorEnv's sub-envs live in other processes and aren't directly
                    # callable here, but this way progress is visible either way.
                    pos_str = "  ".join(f"env{i}: {current_posx[i]:6.2f}" for i in range(num_envs))
                    print(f"\rEpoch {epoch}  Step {t + 1}/{local_steps_per_epoch}  {pos_str}", end="", flush=True)

                    obs = next_obs

            print()  # end the \r-overwritten progress line before the epoch summary below

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state(obs_rms)

            data = combine_buffers(bufs)
            data = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
            ac = run_update(ac, data, pi_optimizer, vf_optimizer, train_pi_iters, train_v_iters,
                             clip_ratio, target_kl, logger, 1024, ent_coef, max_grad_norm)

            # Every epoch starts fresh for every env, even mid-episode - matches the
            # original single-env loop, which reset unconditionally on epoch_ended too.
            obs, _ = envs.reset()
            obs = update_and_normalize(obs)
            ep_ret[:] = 0
            ep_len[:] = 0
            ep_max_posx[:] = 0
            ep_sum_posx[:] = 0
            ep_sum_velx[:] = 0

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('EpMaxPosx', with_min_and_max=True)
            logger.log_tabular('EpAvgPosx', with_min_and_max=True)
            logger.log_tabular('EpAvgVelx', with_min_and_max=True)
            logger.log_tabular('EpStrideCount', with_min_and_max=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()
    finally:
        envs.close()
        logger.close()