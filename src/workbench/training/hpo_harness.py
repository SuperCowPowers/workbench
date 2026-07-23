"""Framework-agnostic hyperparameter-search harness for model training scripts.

Training-only (per the :mod:`workbench.training` contract — imports ``optuna`` and,
for the parallel offload, ``ray[tune]``); templates import this **only inside their
``__main__``**.

The harness owns the *search*: it samples a space, runs trials through a backend
(Optuna serial for local runs, Ray Tune + ASHA for the parallel GPU offload),
prunes weak trials early, and returns the best config. It is framework-agnostic —
each model framework supplies a ``trial_fn`` that builds/trains/scores one
candidate plus a default search space (e.g. :mod:`workbench.training.chemprop_hpo`).

``trial_fn(config, report) -> float`` contract:

* build + train the framework model for ``config`` (single-fold),
* call ``report(step=<epoch>, <metric>=<value>)`` each epoch so the backend can
  prune weak trials early (ASHA / successive halving),
* return the final objective value (the same ``<metric>``), which the harness
  minimizes or maximizes per ``mode``.

The search space is expressed with backend-agnostic specs (:class:`IntRange`,
:class:`FloatRange`, :class:`Choice`) that each backend translates to its own
sampler. ``Choice`` options may be unhashable (e.g. a tapered ``ffn_hidden_dim``
list like ``[1024, 256, 64]``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence, Union

log = logging.getLogger("workbench")


# --- backend-agnostic search-space specs -----------------------------------


@dataclass(frozen=True)
class IntRange:
    """Integer knob sampled in ``[low, high]`` on a ``step`` grid."""

    low: int
    high: int
    step: int = 1


@dataclass(frozen=True)
class FloatRange:
    """Float knob in ``[low, high]``. ``log`` samples log-uniformly; ``step`` (linear
    only) quantizes to a grid."""

    low: float
    high: float
    step: Union[float, None] = None
    log: bool = False


@dataclass(frozen=True)
class Choice:
    """Categorical knob. ``options`` may include unhashable values (e.g. lists)."""

    options: Sequence


Spec = Union[IntRange, FloatRange, Choice]
SearchSpace = dict

# Pruning grace. Successive-halving/ASHA defaults judge a trial almost immediately,
# which wrecks a small search (tens of trials): configs get killed before they've
# trained enough to rank. No trial is eligible for pruning until it has reported this
# many steps (epochs), and Optuna additionally needs this many completed trials as
# baselines before it prunes anything.
PRUNE_WARMUP_STEPS = 20
PRUNE_STARTUP_TRIALS = 5


@dataclass
class HpoResult:
    """Outcome of a search: the winning config plus a record of every trial."""

    best_config: dict
    best_value: float
    metric: str
    mode: str
    n_trials: int
    trials: list = field(default_factory=list)


def run_search(
    trial_fn: Callable[..., float],
    search_space: SearchSpace,
    *,
    n_trials: int = 40,
    backend: str = "auto",
    max_parallel: int = 1,
    metric: str = "holdout_mae",
    mode: str = "min",
    pruning: bool = True,
    seed: int = 42,
    resources_per_trial: Union[dict, None] = None,
) -> HpoResult:
    """Search ``search_space`` for the ``trial_fn`` config that best optimizes ``metric``.

    Args:
        trial_fn: ``(config, report) -> float`` — trains one candidate and returns
            its objective value (see module docstring for the ``report`` protocol).
        search_space: ``{name: Spec}`` mapping knob names to :class:`IntRange` /
            :class:`FloatRange` / :class:`Choice`.
        n_trials: search budget (number of candidate configs).
        backend: ``"optuna"`` (serial), ``"ray"`` (parallel + ASHA), or ``"auto"``
            (ray when importable, else optuna).
        max_parallel: concurrent trials (Optuna: thread jobs; Ray: max concurrency).
        metric: the objective key reported by ``trial_fn``/``report``.
        mode: ``"min"`` or ``"max"``.
        pruning: enable early-stopping of weak trials (successive halving / ASHA).
        seed: sampler seed for reproducible searches.
        resources_per_trial: Ray only — e.g. ``{"gpu": 1}`` (one trial per GPU).

    Returns:
        HpoResult: best config/value plus a per-trial record.
    """
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    backend = _resolve_backend(backend)
    if backend not in ("optuna", "ray"):
        raise ValueError(f"backend must be 'optuna', 'ray', or 'auto', got {backend!r}")
    log.info(
        f"HPO search: backend={backend}, n_trials={n_trials}, metric={metric} ({mode}), "
        f"max_parallel={max_parallel}, knobs={list(search_space)}"
    )
    if backend == "ray":
        return _run_ray(
            trial_fn,
            search_space,
            n_trials=n_trials,
            max_parallel=max_parallel,
            metric=metric,
            mode=mode,
            pruning=pruning,
            seed=seed,
            resources_per_trial=resources_per_trial,
        )
    return _run_optuna(
        trial_fn,
        search_space,
        n_trials=n_trials,
        max_parallel=max_parallel,
        metric=metric,
        mode=mode,
        pruning=pruning,
        seed=seed,
    )


def _resolve_backend(backend: str) -> str:
    """Resolve ``"auto"`` to ``"ray"`` when ray is importable, else ``"optuna"``."""
    if backend != "auto":
        return backend
    try:
        import ray  # noqa: F401

        return "ray"
    except ImportError:
        return "optuna"


# --- Optuna backend (local, serial) ----------------------------------------


def _run_optuna(trial_fn, search_space, *, n_trials, max_parallel, metric, mode, pruning, seed) -> HpoResult:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    # MedianPruner rather than successive halving: at our trial budgets, halving prunes
    # before configs are rankable. Median prunes a trial only once baselines exist and it
    # is worse than the median at the same step.
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=PRUNE_STARTUP_TRIALS, n_warmup_steps=PRUNE_WARMUP_STEPS)
        if pruning
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(direction="minimize" if mode == "min" else "maximize", sampler=sampler, pruner=pruner)

    def objective(trial):
        config = _suggest_optuna(trial, search_space)
        # Stash the resolved (real-valued) config so best_config/trials report
        # actual values, not the categorical indices used for unhashable Choices.
        trial.set_user_attr("config", config)

        def report(step=None, **metrics):
            value = metrics.get(metric)
            if value is None or step is None:
                return
            trial.report(value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return trial_fn(config, report)

    study.optimize(objective, n_trials=n_trials, n_jobs=max_parallel)

    trials = [
        {"number": t.number, "value": t.value, "state": t.state.name, "config": t.user_attrs.get("config", {})}
        for t in study.trials
    ]
    best = study.best_trial
    return HpoResult(
        best_config=best.user_attrs.get("config", dict(best.params)),
        best_value=best.value,
        metric=metric,
        mode=mode,
        n_trials=len(study.trials),
        trials=trials,
    )


def _suggest_optuna(trial, search_space) -> dict:
    """Sample one config from ``search_space`` using an Optuna ``trial``."""
    config = {}
    for name, spec in search_space.items():
        if isinstance(spec, IntRange):
            config[name] = trial.suggest_int(name, spec.low, spec.high, step=spec.step)
        elif isinstance(spec, FloatRange):
            if spec.log:
                config[name] = trial.suggest_float(name, spec.low, spec.high, log=True)
            elif spec.step is not None:
                config[name] = trial.suggest_float(name, spec.low, spec.high, step=spec.step)
            else:
                config[name] = trial.suggest_float(name, spec.low, spec.high)
        elif isinstance(spec, Choice):
            # Options may be unhashable (lists) — suggest an index, map back to the value.
            options = list(spec.options)
            idx = trial.suggest_categorical(name, list(range(len(options))))
            config[name] = options[idx]
        else:
            raise TypeError(f"Unknown search spec for {name!r}: {type(spec).__name__}")
    return config


# --- Ray Tune backend (offload, parallel + ASHA) ---------------------------
# Exercised only in a ray-enabled training container (ray is the `training` extra
# and needs a GPU box for real parallelism); the Optuna backend is what CI covers.


def _run_ray(
    trial_fn, search_space, *, n_trials, max_parallel, metric, mode, pruning, seed, resources_per_trial
) -> HpoResult:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    param_space = _to_ray_space(search_space)

    # ASHA advances on this attribute. Reporting the caller's `step` (the model's epoch)
    # under it — rather than leaving Ray's default training_iteration to count report()
    # calls — keeps pruning decisions aligned with the model's notion of progress.
    time_attr = "step"

    def trainable(config):
        last_step = 0

        def report(step=None, **metrics):
            nonlocal last_step
            last_step = step if step is not None else last_step + 1
            tune.report({**metrics, time_attr: last_step})

        value = trial_fn(config, report)
        # Final objective, one tick past the last epoch (time_attr must increase).
        tune.report({metric: value, time_attr: last_step + 1})

    trainable_res = tune.with_resources(trainable, resources_per_trial) if resources_per_trial else trainable
    # grace_period defaults to 1 (prune after a single report) — far too eager; give each
    # trial the same warmup the Optuna path gets.
    scheduler = (
        ASHAScheduler(metric=metric, mode=mode, time_attr=time_attr, grace_period=PRUNE_WARMUP_STEPS)
        if pruning
        else None
    )
    tuner = tune.Tuner(
        trainable_res,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=n_trials,
            max_concurrent_trials=max_parallel,
            search_alg=OptunaSearch(metric=metric, mode=mode, seed=seed),
            scheduler=scheduler,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric=metric, mode=mode)
    trials = [{"number": i, "value": r.metrics.get(metric), "config": r.config} for i, r in enumerate(results)]
    return HpoResult(
        best_config=best.config,
        best_value=best.metrics.get(metric),
        metric=metric,
        mode=mode,
        n_trials=len(trials),
        trials=trials,
    )


def _to_ray_space(search_space) -> dict:
    """Translate backend-agnostic specs to a Ray Tune ``param_space``."""
    from ray import tune

    space = {}
    for name, spec in search_space.items():
        if isinstance(spec, IntRange):
            space[name] = tune.qrandint(spec.low, spec.high, spec.step)
        elif isinstance(spec, FloatRange):
            if spec.log:
                space[name] = tune.loguniform(spec.low, spec.high)
            elif spec.step is not None:
                space[name] = tune.quniform(spec.low, spec.high, spec.step)
            else:
                space[name] = tune.uniform(spec.low, spec.high)
        elif isinstance(spec, Choice):
            space[name] = tune.choice(list(spec.options))
        else:
            raise TypeError(f"Unknown search spec for {name!r}: {type(spec).__name__}")
    return space
