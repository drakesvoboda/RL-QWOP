"""Lightweight replacements for OpenAI spinup utilities."""

import os
import multiprocessing as mp
import socket

import numpy as np
import torch
import torch.distributed as dist
import scipy.signal

try:
    from mpi4py import MPI
    _HAS_MPI = True
except ImportError:
    _HAS_MPI = False


# ============= MPI Utilities (legacy, mostly unused now - see Multiprocessing Utilities below) =============

def proc_id():
    """Get MPI process rank (0 if MPI not available)."""
    if not _HAS_MPI:
        return 0
    return MPI.COMM_WORLD.Get_rank()


def num_procs():
    """Get total number of MPI processes (1 if MPI not available)."""
    if not _HAS_MPI:
        return 1
    return MPI.COMM_WORLD.Get_size()


def mpi_fork(n):
    """No-op. Parallelism now goes through `run_parallel`/`ppo(num_procs=...)` instead of MPI."""
    pass


def mpi_avg(x):
    """Average a value across all MPI processes (no-op if MPI unavailable)."""
    assert isinstance(x, (int, float, np.ndarray))
    if not _HAS_MPI:
        return x

    comm = MPI.COMM_WORLD
    x = np.asarray(x)
    out = np.empty_like(x)
    comm.Allreduce(x, out, op=MPI.SUM)
    out /= comm.Get_size()
    return out if x.shape else out.item()


def mpi_statistics_scalar(x):
    """Compute mean and std of a scalar across all processes."""
    x = np.asarray(x)

    if not _HAS_MPI:
        return np.mean(x), np.std(x)

    comm = MPI.COMM_WORLD

    global_sum = np.sum(x)
    global_count = len(x)

    total_sum = np.zeros(1)
    total_count = np.zeros(1)

    comm.Allreduce([np.array([global_sum]), MPI.DOUBLE],
                   [total_sum, MPI.DOUBLE], op=MPI.SUM)
    comm.Allreduce([np.array([global_count]), MPI.INT],
                   [total_count, MPI.INT], op=MPI.SUM)

    mean = total_sum[0] / total_count[0]

    # Compute variance
    var_sum = np.sum((x - mean) ** 2)
    total_var = np.zeros(1)
    comm.Allreduce([np.array([var_sum]), MPI.DOUBLE],
                   [total_var, MPI.DOUBLE], op=MPI.SUM)

    std = np.sqrt(total_var[0] / total_count[0])
    return mean, std


def setup_pytorch_for_mpi():
    """Setup PyTorch for distributed training with MPI."""
    torch.set_num_threads(1)


def sync_params(module):
    """Broadcast module parameters from rank 0 to every other rank (no-op outside a
    torch.distributed process group, i.e. when running with a single process)."""
    if not dist.is_available() or not dist.is_initialized():
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=0)


def mpi_avg_grads(module):
    """Average gradients across all MPI processes (no-op if unavailable)."""
    if not _HAS_MPI:
        return

    comm = MPI.COMM_WORLD
    for p in module.parameters():
        if p.grad is not None:
            g = p.grad.data.cpu().numpy()
            comm.Allreduce(g, op=MPI.SUM)
            g /= comm.Get_size()
            p.grad.data = torch.from_numpy(g).to(p.device)


# ============= Core RL Utilities =============

def combined_shape(length, shape=None):
    """Compute combined shape for a batch."""
    if shape is None:
        return (length,)
    return (length, shape) if isinstance(shape, int) else (length, *shape)


def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sum.

    [x0, x1, x2] with discount d →
    [x0 + d*x1 + d^2*x2, x1 + d*x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1])[::-1]


def count_vars(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class RunningMeanStd:
    """Tracks a running mean/variance per observation dimension (Welford's parallel
    algorithm) for online observation normalization. Without this, raw observations of
    very different natural scales - torso height (~0.1), joint angles (~+/-pi), joint/
    horizontal velocities (potentially much larger) - get fed straight into a freshly
    initialized network with small random weights, which has no way to treat them
    comparably until it happens to learn per-input scale corrections on its own."""

    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)


# ============= Logging =============

class EpochLogger:
    """Simple logger for tracking training metrics, mirrored to TensorBoard."""

    def __init__(self, **kwargs):
        self.data = {}
        self.output_dir = kwargs.get('output_dir', 'logs')
        self._tabular_keys = []
        self._immediate = {}

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.output_dir)

    def store(self, **kwargs):
        """Store data for current epoch."""
        for k, v in kwargs.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """Register `key` to be printed on the next dump_tabular() call, using whatever
        was store()'d under that key since the last dump - with_min_and_max/average_only
        control which statistics get shown (see dump_tabular). If `val` is given instead,
        it's a value that's already a single number for the epoch (e.g. 'Epoch',
        'TotalEnvInteracts'), printed immediately rather than aggregated from store()."""
        if val is not None:
            print(f"{key}: {val}")
            self._immediate[key] = val
            return
        self._tabular_keys.append((key, with_min_and_max, average_only))

    def dump_tabular(self):
        """Print stats for every key registered via log_tabular() since the last dump:
        avg/std/min/max for with_min_and_max keys, just the average for average_only keys
        (or as a fallback for any key that was store()'d but never registered). Every
        printed value is also written to TensorBoard, using TotalEnvInteracts (falling
        back to Epoch) as the x-axis so runs with different num_envs/steps_per_epoch stay
        comparable on the same wall of samples-seen rather than raw epoch count."""
        step = int(self._immediate.get('TotalEnvInteracts', self._immediate.get('Epoch', 0)))

        printed = set()
        for key, with_min_and_max, average_only in self._tabular_keys:
            v = np.asarray(self.data.get(key, []), dtype=np.float32)
            printed.add(key)
            if v.size == 0:
                continue
            if with_min_and_max and not average_only:
                print(f"{key}: avg={v.mean():.4f} std={v.std():.4f} min={v.min():.4f} max={v.max():.4f}")
                self.writer.add_scalar(f"{key}/avg", v.mean(), step)
                self.writer.add_scalar(f"{key}/std", v.std(), step)
                self.writer.add_scalar(f"{key}/min", v.min(), step)
                self.writer.add_scalar(f"{key}/max", v.max(), step)
            else:
                print(f"{key}: {v.mean():.4f}")
                self.writer.add_scalar(key, v.mean(), step)

        for key, v in self.data.items():
            if key in printed or not v:
                continue
            print(f"{key}: {np.mean(v):.4f}")
            self.writer.add_scalar(key, np.mean(v), step)

        for key, val in self._immediate.items():
            self.writer.add_scalar(key, val, step)
        self.writer.flush()

        self.data = {}
        self._tabular_keys = []
        self._immediate = {}

    def save_config(self, config):
        """Save configuration (no-op for now)."""
        pass

    def log(self, msg, **kwargs):
        """Log a message."""
        print(msg)

    def setup_pytorch_saver(self, model):
        """Remember the model whose weights save_state() checkpoints."""
        self.model = model

    def save_state(self, obs_rms=None):
        """Checkpoint the actor-critic's weights, plus the observation normalizer's
        running stats if one is in use, to <output_dir>/model.pt - overwriting the
        previous checkpoint rather than keeping per-epoch history, so this doesn't grow
        unbounded across a long/open-ended run. obs_rms matters here as much as the
        weights do: the policy was trained on observations normalized by its mean/var,
        so a checkpoint without it can't be correctly replayed later - it'd feed the
        policy raw-scale observations it never saw during training."""
        if not hasattr(self, 'model'):
            return
        os.makedirs(self.output_dir, exist_ok=True)
        checkpoint = {'model_state_dict': self.model.state_dict()}
        if obs_rms is not None:
            checkpoint['obs_rms'] = {'mean': obs_rms.mean, 'var': obs_rms.var, 'count': obs_rms.count}
        path = os.path.join(self.output_dir, 'model.pt')
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def close(self):
        """Flush and close the TensorBoard writer."""
        self.writer.close()


# ============= Multiprocessing Utilities =============
# Replaces MPI: cross-process synchronization (gathering each rank's rollout buffer,
# broadcasting the updated policy) goes through torch.distributed's "gloo" backend, which
# runs over plain local TCP and needs no MPI runtime installed. Workers are still local
# processes forked from this one (not a separate `torchrun`/`mpirun`-style launch), so
# `env_fn` and other closures/lambdas can be passed straight through.

def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_rank(rank, num_procs, port, target, args):
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=num_procs,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        return target(*args)
    finally:
        dist.destroy_process_group()


def run_parallel(num_procs, target, args=()):
    """Run `target(*args)` across `num_procs` processes, including the caller as rank 0.

    When `num_procs > 1`, `target` runs inside a torch.distributed process group (gloo
    backend, so use `torch.distributed.get_rank()`, `.broadcast()`, `.gather()`, etc.
    directly for cross-process communication - no comm object is threaded through).

    Workers are forked (not spawned), so `target`/`args` may hold closures or lambdas that
    aren't picklable. Blocks until every rank finishes; returns rank 0's return value.
    """
    if num_procs <= 1:
        return target(*args)

    ctx = mp.get_context("fork")
    port = _free_tcp_port()
    workers = []

    for rank in range(1, num_procs):
        p = ctx.Process(target=_run_rank, args=(rank, num_procs, port, target, args), daemon=True)
        p.start()
        workers.append(p)

    try:
        result = _run_rank(0, num_procs, port, target, args)
    finally:
        for p in workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    return result
