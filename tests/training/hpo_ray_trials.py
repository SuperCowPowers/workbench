"""Trial functions for the Ray-backend harness tests.

Ray runs each trial in its own worker process, and a worker can only import what is on
its path — the installed ``workbench`` package, not the pytest module that defined a
closure. These live in a plain module so ``tests/training`` on ``PYTHONPATH`` (set by
the ``ray_cluster`` fixture) is enough to reconstruct them.
"""


def quadratic(config):
    """A smooth bowl minimized at x=3.0, depth=4."""
    return (config["x"] - 3.0) ** 2 + (config["depth"] - 4) ** 2


def quadratic_score(config, index):
    """The same bowl under the ``evaluate_configs`` contract — ``(config, index)``."""
    return (config["x"] - 3.0) ** 2 + (config["depth"] - 4) ** 2


def oom_above_depth_3(config):
    """Raises a real ``torch.cuda.OutOfMemoryError`` in the upper half of the range."""
    import torch

    if config["depth"] >= 4:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (synthetic)")
    return float(config["depth"])


def always_oom(config):
    """Every trial dies — the 'no usable trial' case."""
    import torch

    raise torch.cuda.OutOfMemoryError("CUDA out of memory (synthetic)")
