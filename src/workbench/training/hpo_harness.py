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

* build + train the framework model for ``config``,
* call ``report(step=<tick>, <metric>=<value>)`` as the trial progresses so the
  backend can prune weak trials early (ASHA / successive halving). ``step`` is
  whatever unit of progress the framework reports — an epoch, or a completed
  ensemble fold — and ``prune_warmup`` must be set to match that unit,
* return the final objective value (the same ``<metric>``), which the harness
  minimizes or maximizes per ``mode``.

The search space is expressed with backend-agnostic specs (:class:`IntRange`,
:class:`FloatRange`, :class:`Choice`) that each backend translates to its own
sampler. ``Choice`` options may be unhashable (e.g. a custom space with list-valued
knobs).
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence, Union

log = logging.getLogger("workbench")


# --- backend-agnostic search-space specs -----------------------------------


# Every spec carries a ``default`` — where the knob sits when nobody tunes it. Samplers
# ignore it; it is what makes a space self-describing (the range AND the baseline in one
# object) and what keeps the un-overridden knobs out of the search records as real values
# rather than NaN. A default need not lie inside its own range: the search deliberately
# explores away from where the untuned model sits.


@dataclass(frozen=True)
class IntRange:
    """Integer knob sampled in ``[low, high]`` on a ``step`` grid."""

    low: int
    high: int
    step: int = 1
    default: Union[int, None] = None

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(f"IntRange needs low < high, got low={self.low}, high={self.high}")
        if self.step < 1:
            raise ValueError(f"IntRange step must be >= 1, got {self.step}")

    def to_dict(self) -> dict:
        return _spec_dict("int", {"low": self.low, "high": self.high, "step": self.step}, self.default)


@dataclass(frozen=True)
class FloatRange:
    """Float knob in ``[low, high]``. ``log`` samples log-uniformly; ``step`` (linear
    only) quantizes to a grid."""

    low: float
    high: float
    step: Union[float, None] = None
    log: bool = False
    default: Union[float, None] = None

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(f"FloatRange needs low < high, got low={self.low}, high={self.high}")
        if self.log and self.low <= 0:
            raise ValueError(f"FloatRange(log=True) needs low > 0, got low={self.low}")

    def to_dict(self) -> dict:
        fields = {"low": self.low, "high": self.high, "step": self.step, "log": self.log}
        return _spec_dict("float", fields, self.default)


@dataclass(frozen=True)
class Choice:
    """Categorical knob. ``options`` may include unhashable values (e.g. lists)."""

    options: Sequence
    default: object = None

    def __post_init__(self):
        if not len(self.options):
            raise ValueError("Choice needs at least one option")

    def to_dict(self) -> dict:
        return _spec_dict("choice", {"options": list(self.options)}, self.default)


Spec = Union[IntRange, FloatRange, Choice]

# Frameworks with a defined search space. Each module exposes resolve_search_space(),
# which accepts the same shorthand as the hpo["search_space"] key.
SEARCH_SPACE_MODULES = {
    "chemprop": "workbench.training.chemprop_hpo",
    "xgboost": "workbench.training.xgb_hpo",
    "pytorch": "workbench.training.pytorch_hpo",
}

_SPEC_CLASSES = {"int": IntRange, "float": FloatRange, "choice": Choice}


def _spec_dict(dist: str, fields: dict, default) -> dict:
    """A spec's wire form: ``dist`` plus its own fields, dropping anything unset."""
    out = {"dist": dist, **{key: value for key, value in fields.items() if value is not None}}
    if default is not None:
        out["default"] = default
    return out


def spec_from_dict(spec: dict) -> Spec:
    """Build a spec from its wire form. ``dist`` is required — ``low: 1`` versus ``low: 1.0``
    is too thin a signal to infer an int knob from a float one."""
    fields = dict(spec)
    dist = fields.pop("dist", None)
    if dist not in _SPEC_CLASSES:
        raise ValueError(f"search space needs a 'dist' of {sorted(_SPEC_CLASSES)}, got {dist!r}")
    try:
        return _SPEC_CLASSES[dist](**fields)
    except TypeError as e:
        raise ValueError(f"bad fields for a '{dist}' knob: {e}") from e


def _framework_space(framework: str) -> dict:
    """The shipped ``{knob: Spec}`` space for a framework name."""
    if framework not in SEARCH_SPACE_MODULES:
        raise ValueError(f"No HPO search space for framework {framework!r} (have {sorted(SEARCH_SPACE_MODULES)})")

    # Deferred: the framework modules import *from* this one, so a module-level import
    # would be circular. They defer their own optuna/framework imports, which is what
    # lets a lean environment (the dashboard) describe a space without them installed.
    import importlib

    return importlib.import_module(SEARCH_SPACE_MODULES[framework]).resolve_search_space(None)


class SearchSpace(dict):
    """A ``{knob: Spec}`` search space, with JSON in and out.

    Subclasses ``dict`` so a plain dict works everywhere a SearchSpace does — the class is
    an editor, never a requirement. Start from a framework's shipped space, adjust the knobs
    you have an opinion about, and hand the JSON to ``hpo["search_space"]``::

        space = SearchSpace("chemprop")
        space["max_lr"] = FloatRange(1e-4, 1e-2, log=True, default=3e-3)
        del space["depth"]
        fs.to_model(..., hyperparameters={"hpo": {"search_space": space.to_dict()}})

    What you pass is the *whole* space: a one-knob dict searches one knob.

    Args:
        framework (str): ``"chemprop"``, ``"xgboost"``, or ``"pytorch"``.
        knobs (dict): an explicit ``{knob: Spec}`` mapping instead of a framework's.
    """

    def __init__(self, framework: str = None, knobs: dict = None):
        if framework is not None and knobs is not None:
            raise ValueError("SearchSpace takes a framework or knobs, not both")
        self.framework = framework
        super().__init__(_framework_space(framework) if framework is not None else (knobs or {}))

    @classmethod
    def from_dict(cls, spec: dict) -> "SearchSpace":
        """Build from the JSON wire form — ``{knob: {"dist": ..., ...}}``."""
        return cls(knobs={knob: spec_from_dict(fields) for knob, fields in spec.items()})

    def to_dict(self) -> dict:
        """The JSON wire form, suitable for ``hpo["search_space"]``."""
        return {knob: spec.to_dict() for knob, spec in self.items()}

    def to_frame(self):
        """One row per knob: pinned ``knob``/``default``/``dist`` plus a ``spec`` JSON blob
        carrying whatever fields that ``dist`` has."""
        import json

        import pandas as pd

        rows = []
        for knob, spec in self.items():
            fields = spec.to_dict()
            rows.append(
                {
                    "knob": knob,
                    "default": fields.pop("default", None),
                    "dist": fields.pop("dist"),
                    "spec": json.dumps(fields),
                }
            )
        # `default` holds each knob's native type (an int width, a float rate, a shape
        # string), so the column is built as object rather than upcast to float.
        return pd.DataFrame(
            {
                "knob": [row["knob"] for row in rows],
                "default": pd.Series([row["default"] for row in rows], dtype=object),
                "dist": [row["dist"] for row in rows],
                "spec": [row["spec"] for row in rows],
            }
        )

    def subset(self, groups) -> "SearchSpace":
        """Narrow to named knob groups (``"basic"``, ``"basic+optimizer"``)."""
        if self.framework is None:
            raise ValueError("subset() needs a framework-built SearchSpace")
        import importlib

        module = importlib.import_module(SEARCH_SPACE_MODULES[self.framework])
        return SearchSpace(knobs=module.resolve_search_space(groups))


def space_defaults(search_space: SearchSpace) -> dict:
    """The ``{knob: default}`` baseline a space describes — the untuned config it searches around."""
    return {knob: spec.default for knob, spec in search_space.items()}


# Pruning grace. Successive-halving/ASHA defaults judge a trial almost immediately,
# which wrecks a small search (tens of trials): configs get killed before they've
# trained enough to rank. No trial is eligible for pruning until it has reported this
# many steps, and Optuna additionally needs this many completed trials as baselines
# before it prunes anything. The default suits per-epoch reporters; callers reporting
# a coarser step (e.g. one per ensemble fold) pass a smaller ``prune_warmup``.
PRUNE_WARMUP_STEPS = 20
PRUNE_STARTUP_TRIALS = 5

# Each ASHA cull keeps the top ``1 / reduction_factor``. Ray's default of 4 discards 75% of
# trials on the evidence of two folds, which is most of the search budget spent on one noisy
# comparison; halving keeps 50% instead.
PRUNE_REDUCTION_FACTOR = 2


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
    n_trials: int = 60,
    backend: str = "auto",
    max_parallel: int = 1,
    metric: str = "holdout_mae",
    mode: str = "min",
    pruning: bool = True,
    prune_warmup: int = PRUNE_WARMUP_STEPS,
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
        prune_warmup: steps a trial must report before it is eligible for pruning —
            in the same unit ``trial_fn`` reports (epochs, folds, ...).
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
            prune_warmup=prune_warmup,
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
        prune_warmup=prune_warmup,
        seed=seed,
    )


def evaluate_configs(
    eval_fn: Callable[..., float],
    configs: Sequence[dict],
    *,
    backend: str = "auto",
    max_parallel: int = 1,
    resources_per_trial: Union[dict, None] = None,
) -> list:
    """Score a fixed list of configs — no sampling, no pruning.

    The counterpart to :func:`run_search`, for a confirmation/re-rank pass: every config
    runs to completion, so the returned values are directly comparable to each other in a
    way the search's own (pruned, selection-biased) values are not.

    Args:
        eval_fn: ``(config, index) -> float`` — scores one config.
        configs: the configs to score.
        backend: ``"ray"`` fans them out concurrently, ``"optuna"`` scores them serially
            in-process, ``"auto"`` picks ray when importable.
        max_parallel: concurrent evaluations (Ray only).
        resources_per_trial: Ray only — e.g. ``{"gpu": 1}``.

    Returns:
        list: one value per config, positionally aligned, ``None`` where scoring failed.
    """
    configs = list(configs)
    if not configs:
        return []
    if len(configs) > 1 and _resolve_backend(backend) == "ray":
        return _evaluate_ray(eval_fn, configs, max_parallel=max_parallel, resources_per_trial=resources_per_trial)

    values = []
    for index, config in enumerate(configs):
        try:
            values.append(float(eval_fn(config, index)))
        except Exception as exc:
            log.warning(f"Config {index} failed to evaluate: {exc}")
            values.append(None)
    return values


def _ray_session(func):
    """Give the wrapped function its own Ray session, torn down when it returns.

    Tune's actor manager does not survive a second ``Tuner`` in a session it did not start:
    scheduling races the previous run's teardown and lands on an actor already dropped from
    the manager's tables, which raises "Tracked actor is not managed by this event manager".
    A session per run keeps those tables from outliving the run that filled them.

    Trials run in their own worker processes, so the allocator setting has to travel in the
    runtime env rather than the driver's ``os.environ``. Expandable segments let the caching
    allocator grow a segment instead of demanding one contiguous block, which is what
    fragments when several trials share a card.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import ray

        ray.init(
            runtime_env={"env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}},
            ignore_reinit_error=True,
        )
        try:
            return func(*args, **kwargs)
        finally:
            ray.shutdown()

    return wrapper


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


def _run_optuna(trial_fn, search_space, *, n_trials, max_parallel, metric, mode, pruning, prune_warmup, seed):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    # MedianPruner rather than successive halving: at our trial budgets, halving prunes
    # before configs are rankable. Median prunes a trial only once baselines exist and it
    # is worse than the median at the same step.
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=PRUNE_STARTUP_TRIALS, n_warmup_steps=prune_warmup)
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

    # Serial: n_jobs>1 runs trials on threads in one process, racing pl.seed_everything's
    # global RNG and contending on the single GPU. Real parallelism is the Ray offload's job.
    if max_parallel > 1:
        log.info(f"Optuna backend is serial; ignoring max_parallel={max_parallel} (use backend='ray' for parallel).")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    trials = [
        {"number": t.number, "value": t.value, "state": t.state.name, "config": t.user_attrs.get("config", {})}
        for t in study.trials
    ]
    # Rank explicitly rather than via study.best_trial: its "No trials are completed yet"
    # gives no clue what went wrong, and the most likely cause has a specific fix — Optuna
    # marks a trial FAIL when the objective returns NaN, so an unlabeled target column fails
    # every trial and only shows up here, after the whole search has been paid for.
    completed = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
    if not completed:
        states = {}
        for t in study.trials:
            states[t.state.name] = states.get(t.state.name, 0) + 1
        raise RuntimeError(
            f"HPO search produced no usable trial (states: {states}). If trials FAILed, a NaN "
            "objective is the usual cause — check the target column has non-NaN values."
        )
    best = min(completed, key=lambda t: t.value) if mode == "min" else max(completed, key=lambda t: t.value)
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


# The fence is a fraction of the card's *total* memory, so co-tenant trials whose shares sum
# to 1.0 leave nothing for their CUDA contexts (a few hundred MB each, allocated by the driver
# outside the caching allocator). Scaling every share down keeps that space free.
_FENCE_HEADROOM = 0.9


def _is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory error. False when torch isn't installed.

    Called from an ``except`` block, so it must not raise: anything it raises would replace
    the exception being handled. ``OutOfMemoryError`` only exists on torch >= 1.13.
    """
    try:
        import torch
    except ImportError:
        return False
    oom_error = getattr(torch.cuda, "OutOfMemoryError", None)
    return oom_error is not None and isinstance(exc, oom_error)


def _fence_gpu_memory(resources_per_trial) -> None:
    """Cap this trial process at the share of the GPU it was scheduled for.

    ``gpus_per_trial`` is only a *placement* hint — Ray packs two trials onto a card, but
    nothing stops one of them allocating the whole thing and OOMing its neighbour. Fencing
    each process makes an oversized config fail on its own first oversized allocation, so
    the failure is attributable to the trial that caused it instead of the ones behind it.

    Ray gives each trial its own ``CUDA_VISIBLE_DEVICES``, so device 0 is this trial's card.
    A no-op when the trial holds no GPU (the CPU-resourced XGBoost path) or torch is absent.
    """
    share = (resources_per_trial or {}).get("gpu")
    if not share:
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        # Headroom covers the *co-tenant* CUDA contexts the fraction cannot see — the driver
        # allocates them outside the caching allocator, so shares summing to 1.0 would leave
        # them nowhere to live. A trial holding the whole card has only its own context and
        # needs no such reserve; taking it anyway would shrink the multi-task path, which
        # already sits near the card's limit.
        fraction = float(share)
        if fraction < 1.0:
            fraction *= _FENCE_HEADROOM
        torch.cuda.set_per_process_memory_fraction(min(1.0, fraction), 0)


def _partial_aware_search(done_flag: str, **kwargs):
    """An ``OptunaSearch`` that tells Optuna a scheduler-stopped trial is *pruned*.

    Ray ends a stopped trial by handing the searcher that trial's last result, and
    ``OptunaSearch`` records any result carrying the metric as a COMPLETE observation. For
    an ensemble objective that last value is a *partial* ensemble — systematically worse
    than the full one — so TPE's quantile split ends up fitting a mixture of two different
    objectives, with every pruned region labelled worse than it is. That entrenches whatever
    the first rung decided.

    Dropping the result for any trial that never set ``done_flag`` maps it to PRUNED, which
    TPE ranks by the intermediate values it already recorded — the comparison ASHA actually
    made — instead of by a value on the wrong scale. Winner selection guards the same
    distinction independently, on the harness's own records.
    """
    from ray.tune.search.optuna import OptunaSearch

    class _PartialAware(OptunaSearch):
        def on_trial_complete(self, trial_id, result=None, error=False):
            if result is not None and not result.get(done_flag):
                result = None
            super().on_trial_complete(trial_id, result=result, error=error)

    return _PartialAware(**kwargs)


def _resolve_choices(config, choice_options) -> dict:
    """Map ``Choice`` knobs back from the indices the trainable sampled them as."""
    return {k: (choice_options[k][v] if k in choice_options else v) for k, v in (config or {}).items()}


def _scored_value(result, metric):
    """A trial's objective, or None when it never produced a usable one.

    An OOM'd trial reports NaN so the scheduler's strict metric check is satisfied. That NaN
    must not survive into rankings or records: it compares as neither better nor worse than
    anything, so `min()` would seat it wherever iteration order happened to put it.
    """
    value = (getattr(result, "metrics", None) or {}).get(metric)
    return None if value is None or value != value else value  # value != value catches NaN


def _resolve_trial_records(results, *, metric, done_flag, choice_options) -> list:
    """One record per trial, defensive about trials that never reported.

    A trial killed before it ran — an OOM-killed worker, a missing dependency in the image,
    actor construction failing — comes back with ``config`` and ``metrics`` unset. Those
    still belong in the record as the unscored trials they are; reading them unguarded
    discards a search that has already been paid for in full.
    """
    return [
        {
            "number": i,
            "value": _scored_value(r, metric),
            "config": _resolve_choices(r.config, choice_options),
            "completed": bool((r.metrics or {}).get(done_flag)),
        }
        for i, r in enumerate(results)
    ]


@_ray_session
def _run_ray(
    trial_fn, search_space, *, n_trials, max_parallel, metric, mode, pruning, prune_warmup, seed, resources_per_trial
) -> HpoResult:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    # Choice knobs are sampled as an index and mapped back to the value here (mirroring the
    # Optuna path): OptunaSearch's categorical rejects unhashable options (list-valued
    # knobs), so passing the raw list only works by warning-and-degrading.
    param_space, choice_options = _to_ray_space(search_space)

    # ASHA advances on this attribute. Reporting the caller's `step` (the model's epoch)
    # under it — rather than leaving Ray's default training_iteration to count report()
    # calls — keeps pruning decisions aligned with the model's notion of progress.
    time_attr = "step"
    done_flag = "_hpo_completed"

    def trainable(config):
        _fence_gpu_memory(resources_per_trial)
        config = {k: (choice_options[k][v] if k in choice_options else v) for k, v in config.items()}
        last_step = 0

        def report(step=None, **metrics):
            nonlocal last_step
            last_step = step if step is not None else last_step + 1
            tune.report({**metrics, time_attr: last_step})

        try:
            value = trial_fn(config, report)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            # Letting the OOM escape gets the trial reported to Optuna as FAIL, and TPE draws
            # only on COMPLETE and PRUNED — so the sampler learns nothing and keeps proposing
            # a corner that cannot fit. Reporting NaN under the metric ends the trial without
            # a usable score while keeping every consumer happy:
            #
            # * the scheduler's strict metric check needs the key present. Returning nothing
            #   is only exempt when the result carries no other keys, and Ray injects the
            #   flattened `config/*` into it — so an unreported trial takes down the tuner.
            # * `nanpercentile` skips NaN, so a dead trial cannot move a rung's cutoff, and
            #   `nan < cutoff` is False, so it is never mistaken for a prunable live trial.
            # * `_partial_aware_search` drops the result for want of `done_flag`, so Optuna is
            #   told PRUNED with no value — never a NaN objective, which it rejects.
            #
            # None is not an option in its place: Ray hands the metric straight to
            # `optuna.Trial.report`, which requires a float.
            log.warning(f"Trial out of GPU memory, ending it unscored: {exc}")
            tune.report({metric: float("nan"), time_attr: last_step + 1})
            return
        # Final objective, one tick past the last epoch (time_attr must increase). done_flag
        # marks a full run so winner selection can exclude ASHA-stopped partial ensembles.
        tune.report({metric: value, time_attr: last_step + 1, done_flag: 1})

    trainable_res = tune.with_resources(trainable, resources_per_trial) if resources_per_trial else trainable
    # grace_period defaults to 1 (prune after a single report) — far too eager; give each
    # trial the same warmup the Optuna path gets.
    # One rung, at prune_warmup. Rungs sit at grace_period * reduction_factor ** k, so a 50%
    # cull also buys rungs at 4, 8, ... — a bad trade here, since a trial reaching fold 4 of 5
    # has paid 80% of its cost and stopping it saves one fold while forfeiting its place in the
    # ranking pool. max_t bounds the ladder and, with stop_last_trials off, does nothing else.
    scheduler = (
        ASHAScheduler(
            metric=metric,
            mode=mode,
            time_attr=time_attr,
            grace_period=prune_warmup,
            reduction_factor=PRUNE_REDUCTION_FACTOR,
            max_t=prune_warmup + 1,
            stop_last_trials=False,
        )
        if pruning
        else None
    )
    tuner = tune.Tuner(
        trainable_res,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=n_trials,
            max_concurrent_trials=max_parallel,
            search_alg=_partial_aware_search(done_flag, metric=metric, mode=mode, seed=seed),
            scheduler=scheduler,
        ),
    )
    results = tuner.fit()

    # Rank only among trials that ran the full ensemble. An ASHA-stopped trial's last value
    # is a *partial* ensemble mean, which can read lower than a completed one and would
    # otherwise be published — the very regime mismatch the ensemble objective removes.
    completed = [r for r in results if (r.metrics or {}).get(done_flag)]
    # Fall back to any *scored* trial if pruning stopped everything. Never to an unscored
    # one: a trial that died carries neither a value nor a config, so ranking it would
    # publish `None` as the winning config.
    pool = completed or [r for r in results if _scored_value(r, metric) is not None]
    if not pool:
        # Mirror the Optuna path's actionable failure. Ray otherwise surfaces this as an
        # AttributeError deep in config resolution, on the box that costs the most to rent.
        errored = sum(1 for r in results if r.error is not None)
        raise RuntimeError(
            f"HPO search produced no usable trial ({len(results)} trials, {errored} errored). "
            "If trials errored, check the logs above for the first traceback — an OOM or a "
            "NaN objective (e.g. an unlabeled target column) is the usual cause."
        )

    def _objective(r):
        # Missing/None metric sorts to the worst end regardless of mode — a signed default
        # (e.g. -inf in max mode) would otherwise make an unscored trial win.
        v = _scored_value(r, metric)
        if v is None:
            return float("inf")
        return v if mode == "min" else -v

    best = min(pool, key=_objective)
    trials = _resolve_trial_records(results, metric=metric, done_flag=done_flag, choice_options=choice_options)
    return HpoResult(
        best_config=_resolve_choices(best.config, choice_options),
        best_value=_scored_value(best, metric),
        metric=metric,
        mode=mode,
        n_trials=len(trials),
        trials=trials,
    )


@_ray_session
def _evaluate_ray(eval_fn, configs, *, max_parallel, resources_per_trial):
    """Fan :func:`evaluate_configs` out across the node's GPUs via Ray."""
    from ray import tune

    # Configs are captured in the closure and addressed by index — they can hold unhashable
    # values (a list-valued knob) that a Ray param_space would reject.
    idx_key, value_key = "_config_index", "value"

    def trainable(slot):
        index = slot[idx_key]
        tune.report({value_key: float(eval_fn(configs[index], index)), idx_key: index})

    trainable_res = tune.with_resources(trainable, resources_per_trial) if resources_per_trial else trainable
    tuner = tune.Tuner(
        trainable_res,
        # grid_search over the indices runs each config exactly once; num_samples multiplies it.
        param_space={idx_key: tune.grid_search(list(range(len(configs))))},
        tune_config=tune.TuneConfig(num_samples=1, max_concurrent_trials=max_parallel),
    )
    values = [None] * len(configs)
    for result in tuner.fit():
        index = (result.metrics or {}).get(idx_key)
        if index is not None:
            values[int(index)] = result.metrics.get(value_key)
    return values


def _to_ray_space(search_space):
    """Translate backend-agnostic specs to a Ray Tune ``param_space``.

    Returns ``(param_space, choice_options)`` — ``choice_options`` maps each ``Choice``
    knob to its option list, because those knobs are sampled as an *index* (Ray's caller
    unwraps them). This keeps unhashable options (list-valued knobs) out of Optuna's
    categorical, which only accepts hashables.
    """
    from ray import tune

    space, choice_options = {}, {}
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
            options = list(spec.options)
            choice_options[name] = options
            space[name] = tune.choice(list(range(len(options))))
        else:
            raise TypeError(f"Unknown search spec for {name!r}: {type(spec).__name__}")
    return space, choice_options
