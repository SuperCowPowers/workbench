"""Trial functions for the Ray-backend harness tests.

Ray runs each trial in its own worker process, and a worker can only import what is on
its path — the installed ``workbench`` package, not the pytest module that defined a
closure. These live in a plain module so ``tests/training`` on ``PYTHONPATH`` (set by
the ``ray_cluster`` fixture) is enough to reconstruct them.
"""


def quadratic(config, report):
    """A smooth bowl minimized at x=3.0, depth=4; reports one intermediate step."""
    value = (config["x"] - 3.0) ** 2 + (config["depth"] - 4) ** 2
    report(step=1, holdout_mae=value)
    return value


def quadratic_score(config, index):
    """The same bowl under the ``evaluate_configs`` contract — ``(config, index)``, no report."""
    return (config["x"] - 3.0) ** 2 + (config["depth"] - 4) ** 2


def oom_above_depth_3(config, report):
    """Raises a real ``torch.cuda.OutOfMemoryError`` in the upper half of the range."""
    import torch

    if config["depth"] >= 4:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (synthetic)")
    return float(config["depth"])


def always_oom(config, report):
    """Every trial dies — the 'no usable trial' case."""
    import torch

    raise torch.cuda.OutOfMemoryError("CUDA out of memory (synthetic)")


def oom_before_any_report(config, report):
    """OOM on the first fold, before the trial has reported anything.

    The case that matters: a trial that finishes without ever reporting the metric trips the
    scheduler's strict metric check, which raises out of the tuner and takes the whole search
    with it.
    """
    import torch

    if config["depth"] >= 4:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (synthetic)")
    for step in (1, 2, 3):
        report(step=step, holdout_mae=float(config["depth"]))
    return float(config["depth"])


def five_step_objective(config, report):
    """Reports five steps, as a five-fold ensemble trial does."""
    value = float(config["depth"])
    for step in range(1, 6):
        report(step=step, holdout_mae=value)
    return value
