"""Lightweight replacements for OpenAI spinup utilities."""

import numpy as np
import torch
import scipy.signal

try:
    from mpi4py import MPI
    _HAS_MPI = True
except ImportError:
    _HAS_MPI = False


# ============= MPI Utilities =============

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
    """Fork n processes (no-op if already in MPI context or MPI unavailable)."""
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
    """Synchronize module parameters across all MPI processes (no-op if unavailable)."""
    if not _HAS_MPI:
        return

    comm = MPI.COMM_WORLD
    for p in module.parameters():
        comm.Bcast(p.data.cpu().numpy(), root=0)


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


# ============= Logging =============

class EpochLogger:
    """Simple logger for tracking training metrics."""

    def __init__(self, **kwargs):
        self.data = {}
        self.output_dir = kwargs.get('output_dir', 'logs')

    def store(self, **kwargs):
        """Store data for current epoch."""
        for k, v in kwargs.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def log_tabular(self, key, val=None, **kwargs):
        """Log a tabular value."""
        if val is not None:
            print(f"{key}: {val}")

    def dump_tabular(self):
        """Print logged data."""
        for k, v in self.data.items():
            if v:
                mean_val = np.mean(v)
                print(f"{k}: {mean_val:.4f}")
        self.data = {}

    def save_config(self, config):
        """Save configuration (no-op for now)."""
        pass

    def log(self, msg, **kwargs):
        """Log a message."""
        print(msg)

    def setup_pytorch_saver(self, model):
        """Setup model saver (no-op for now)."""
        self.model = model

    def save_state(self, state_dict, val_fn):
        """Save state (no-op for now)."""
        pass
