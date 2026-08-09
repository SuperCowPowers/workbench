"""Framework-agnostic hyperparameter-search harness for model training scripts.

Training-only (per the :mod:`workbench.training` contract — imports ``optuna`` and,
for the parallel offload, ``ray[tune]``); templates import this **only inside their
``__main__``**.

The harness owns the *search*: it samples a space, runs trials through a backend
(Optuna serial for local runs, Ray Tune for the parallel GPU offload), and returns
the best config. It is framework-agnostic —
each model framework supplies a ``trial_fn`` that builds/trains/scores one
candidate plus a default search space (e.g. :mod:`workbench.training.chemprop_hpo`).

``trial_fn(config, report) -> float`` contract: build + train the framework model for
``config`` and return its objective value, which the harness minimizes or maximizes per
``mode``. A trial built from parts (an ensemble member per step) calls
``report(step, value)`` after each one with the running objective; the backend can then
stop a trial that is already off the pace (successive halving). Only trials that reach
``max_steps`` are ranked — a partial objective is measured on less data than a full one.

The search space is expressed with backend-agnostic specs (:class:`IntRange`,
:class:`FloatRange`, :class:`Choice`) that each backend translates to its own
sampler. ``Choice`` options may be unhashable (e.g. a custom space with list-valued
knobs).
"""

from __future__ import annotations

import functools
import logging
import math
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
    metric: str = "objective",
    mode: str = "min",
    seed: int = 42,
    resources_per_trial: Union[dict, None] = None,
    points_to_evaluate: Union[Sequence[dict], None] = None,
    max_steps: Union[int, None] = None,
) -> HpoResult:
    """Search ``search_space`` for the ``trial_fn`` config that best optimizes ``metric``.

    Args:
        trial_fn: ``(config, report) -> float`` — trains one candidate and returns its
            objective value, calling ``report(step, value)`` as partial results firm up.
        search_space: ``{name: Spec}`` mapping knob names to :class:`IntRange` /
            :class:`FloatRange` / :class:`Choice`.
        n_trials: search budget (number of candidate configs).
        backend: ``"optuna"`` (serial), ``"ray"`` (parallel), or ``"auto"`` (ray when
            importable, else optuna).
        max_parallel: concurrent trials (Optuna: thread jobs; Ray: max concurrency).
        metric: the objective key ``trial_fn`` optimizes.
        mode: ``"min"`` or ``"max"``.
        seed: sampler seed for reproducible searches.
        resources_per_trial: Ray only — e.g. ``{"gpu": 1}`` (one trial per GPU).
        points_to_evaluate: configs to try first, ahead of anything the sampler proposes.
            They count against ``n_trials`` and can win, and the sampler learns from them.
            They are left out of the ladder in both directions -- never culled at a rung,
            and never counted in one either, since a full-fidelity value has no business
            setting the bar for trials measured on fewer folds.
        max_steps: how many ``report`` steps a full trial makes. Setting it turns on
            successive halving over those steps and restricts ranking to trials that reach
            the last one. None runs every trial to term and ranks them all.

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
            seed=seed,
            resources_per_trial=resources_per_trial,
            points_to_evaluate=points_to_evaluate,
            max_steps=max_steps,
        )
    return _run_optuna(
        trial_fn,
        search_space,
        n_trials=n_trials,
        max_parallel=max_parallel,
        metric=metric,
        mode=mode,
        seed=seed,
        points_to_evaluate=points_to_evaluate,
        max_steps=max_steps,
    )


# Each cull keeps the top 1/RUNG_FACTOR of the trials that reached a rung, and rungs sit at
# RUNG_FACTOR**k steps. Two is the gentle end of the usual range: the first rung is where the
# partial estimate is least trustworthy, and a trial killed there is killed silently — no
# later measurement can tell us it should have survived.
RUNG_FACTOR = 2

# The step counter a laddered trial reports alongside its objective; ASHA advances on it.
_STEP = "step"


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


def _is_seeded(config, seeded) -> bool:
    """Whether a trial is one of the enqueued points.

    Subset, not equality: a point may name only some knobs and leave the sampler to fill the
    rest, which is what ``enqueue_trial``/``points_to_evaluate`` accept. An empty point names
    nothing and so seeds nothing — matching it against every trial would switch the ladder
    off entirely.
    """
    return any(point and all(config.get(knob) == value for knob, value in point.items()) for point in seeded)


def _finite(value) -> bool:
    """True for a real number — not None, not NaN, not an infinity."""
    return value is not None and math.isfinite(value)


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


def _run_optuna(
    trial_fn, search_space, *, n_trials, max_parallel, metric, mode, seed, points_to_evaluate=None, max_steps=None
):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    # Mirrors the Ray path's ASHA so the mechanics are exercised by CI, which only runs this
    # backend. Rungs land at the same steps and each keeps the same fraction.
    pruner = (
        optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=RUNG_FACTOR)
        if max_steps
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(direction="minimize" if mode == "min" else "maximize", sampler=sampler, pruner=pruner)
    # Both backends sample a Choice as an index (see _suggest_optuna), so a seeded point has
    # to be expressed the same way.
    choice_options = {name: list(spec.options) for name, spec in search_space.items() if isinstance(spec, Choice)}
    for point in points_to_evaluate or []:
        study.enqueue_trial(_encode_point(point, choice_options), skip_if_exists=True)

    seeded = list(points_to_evaluate or [])

    def objective(trial):
        config = _suggest_optuna(trial, search_space)
        # Stash the resolved (real-valued) config so best_config/trials report
        # actual values, not the categorical indices used for unhashable Choices.
        trial.set_user_attr("config", config)
        laddered = bool(max_steps) and not _is_seeded(config, seeded)

        def report(step, value):
            # A non-finite value is an absence of measurement (sparse multi-target data can
            # leave an early fold with no labelled rows), so there is nothing to record.
            if not _finite(value):
                return
            trial.report(value, step)
            # Optuna files a trial into the rung's competing pool inside `prune()`, so a
            # seeded point that is never asked is neither culled nor counted. The step guard
            # is for a rung landing on the last step (max_steps of 1, 2 or 4), where stopping
            # would discard a trial that already has the full objective.
            if laddered and step < max_steps and trial.should_prune():
                raise optuna.TrialPruned()

        return trial_fn(config, report)

    # Serial: n_jobs>1 runs trials on threads in one process, racing pl.seed_everything's
    # global RNG and contending on the single GPU. Real parallelism is the Ray offload's job.
    if max_parallel > 1:
        log.info(f"Optuna backend is serial; ignoring max_parallel={max_parallel} (use backend='ray' for parallel).")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    trials = [_optuna_record(t, max_steps) for t in study.trials]
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


def _optuna_record(trial, max_steps) -> dict:
    """One trial's record, keyed off its state rather than its value.

    Optuna fills in ``value`` for a PRUNED trial from its last intermediate, so the value
    alone cannot say whether a trial finished. The three outcomes differ in both fields:

    * COMPLETE — ran every step.
    * PRUNED — stopped at a rung; keeps the partial objective it reached, and the step says
      where it stopped.
    * FAIL — no objective at all. It may still have reported before failing, but keeping
      that value would file a genuine failure as a scheduler stop.

    ``trajectory`` is the whole rung history: every trial reports at every step, including
    a seeded point, whose exemption is that the pruner is never consulted about it.
    """
    reported = getattr(trial, "intermediate_values", None) or {}
    state = trial.state.name
    complete = state == "COMPLETE"
    trajectory = {int(step): float(value) for step, value in reported.items() if _finite(value)}
    return {
        "number": trial.number,
        "value": trial.value if state != "FAIL" else None,
        "state": state,
        "config": trial.user_attrs.get("config", {}),
        # The trajectory's endpoint, so the two always agree. The fallback covers an
        # objective that reports nothing: a completed one still ran every step.
        "step": max(reported, default=max_steps if complete else None),
        "trajectory": trajectory,
    }


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


# --- Ray Tune backend (offload, parallel) ----------------------------------
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


def _encode_point(point, choice_options) -> dict:
    """Express a config in the coordinates the sampler searches — Choice knobs as indices.

    The inverse of :func:`_resolve_choices`, for handing the sampler a specific config to
    try. A Choice value the space does not offer cannot be expressed as an index, so it
    raises here rather than seeding the search with a silently different config.
    """
    encoded = {}
    for knob, value in point.items():
        options = choice_options.get(knob)
        if options is None:
            encoded[knob] = value
        elif value in options:
            encoded[knob] = options.index(value)
        else:
            raise ValueError(
                f"cannot seed the search with {knob}={value!r}: it is not one of the searched "
                f"options {options}. Add it to the knob's Choice, or leave {knob} unset."
            )
    return encoded


def _resolve_choices(config, choice_options) -> dict:
    """Map ``Choice`` knobs back from the indices the trainable sampled them as."""
    return {k: (choice_options[k][v] if k in choice_options else v) for k, v in (config or {}).items()}


def _scored_value(result, metric):
    """A trial's objective, or None when it never produced a usable one.

    An OOM'd trial reports NaN rather than dying, so the sampler records it as an
    unpromising region instead of learning nothing. That NaN must not survive into rankings
    or records: it compares as neither better nor worse than anything, so `min()` would seat
    it wherever iteration order happened to put it.
    """
    value = (getattr(result, "metrics", None) or {}).get(metric)
    return None if value is None or value != value else value  # value != value catches NaN


def _reached_full(result, max_steps) -> bool:
    """Whether a trial ran every step, rather than being culled at a rung.

    Only these are comparable: a stopped trial's objective is measured over the steps it
    got through, which is less data than a full trial saw.
    """
    if not max_steps:
        return True
    return ((getattr(result, "metrics", None) or {}).get(_STEP) or 0) >= max_steps


def _laddered_scheduler(*, max_steps, metric, mode, is_exempt):
    """ASHA that neither judges a seeded point at a rung nor lets one set the bar at a rung.

    A seeded point runs at full fidelity, so its value covers every fold while the trials it
    would be measured against have covered fewer. ``on_trial_complete`` is the second half:
    the base class files a finished trial's value at the highest rung it is not already
    recorded in, which for an exempt trial posts a full-fidelity number as a partial cutoff.

    The exemption is here rather than in the trial's ``report``, so a seeded trial still
    leaves the same per-fold record as any other. Both overrides return before the base
    class runs, which leaves the exempt trial's bracket entry behind: one dict entry per
    seeded point, and the price of touching no private state.
    """
    from ray.tune.schedulers import ASHAScheduler, TrialScheduler

    class _ExemptSeeded(ASHAScheduler):
        def on_trial_result(self, tune_controller, trial, result):
            if is_exempt(trial.config):
                return TrialScheduler.CONTINUE
            return super().on_trial_result(tune_controller, trial, result)

        def on_trial_complete(self, tune_controller, trial, result):
            if is_exempt(trial.config):
                return
            super().on_trial_complete(tune_controller, trial, result)

    return _ExemptSeeded(
        time_attr=_STEP,
        metric=metric,
        mode=mode,
        grace_period=1,
        reduction_factor=RUNG_FACTOR,
        max_t=max_steps,
    )


def _ray_trajectory(result, metric) -> dict:
    """A trial's rung history, ``{step: objective}``, from the results Ray kept per report."""
    frame = getattr(result, "metrics_dataframe", None)
    if frame is None or _STEP not in frame or metric not in frame:
        return {}
    pairs = frame[[_STEP, metric]].dropna()
    return {int(step): float(value) for step, value in pairs.itertuples(index=False)}


def _resolve_trial_records(results, *, metric, choice_options, max_steps) -> list:
    """One record per trial, defensive about trials that never reported.

    A trial killed before it ran — an OOM-killed worker, a missing dependency in the image,
    actor construction failing — comes back with ``config`` and ``metrics`` unset. Those
    still belong in the record as the unscored trials they are; reading them unguarded
    discards a search that has already been paid for in full.

    ``completed`` means ranked-eligible: it scored *and* ran every step. A trial stopped at
    a rung keeps its partial value but is not completed, which is what separates "stopped
    early" from "died" downstream. A trial that *raised* keeps no value even if it reported
    before raising — otherwise a genuine failure files itself as a scheduler stop.
    """
    return [
        {
            "number": i,
            "value": None if getattr(r, "error", None) is not None else value,
            "config": _resolve_choices(getattr(r, "config", None), choice_options),
            "step": (getattr(r, "metrics", None) or {}).get(_STEP),
            "completed": value is not None and getattr(r, "error", None) is None and _reached_full(r, max_steps),
            "trajectory": _ray_trajectory(r, metric),
        }
        for i, r in enumerate(results)
        for value in [_scored_value(r, metric)]
    ]


@_ray_session
def _run_ray(
    trial_fn,
    search_space,
    *,
    n_trials,
    max_parallel,
    metric,
    mode,
    seed,
    resources_per_trial,
    points_to_evaluate=None,
    max_steps=None,
) -> HpoResult:
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch

    # Choice knobs are sampled as an index and mapped back to the value here (mirroring the
    # Optuna path): OptunaSearch's categorical rejects unhashable options (list-valued
    # knobs), so passing the raw list only works by warning-and-degrading.
    param_space, choice_options = _to_ray_space(search_space)

    last_step = max_steps or 1
    # The sampler and the scheduler both see a config in the coordinates the search samples,
    # so the seeded points are matched there rather than in resolved values.
    seeded = [_encode_point(point, choice_options) for point in points_to_evaluate or []]

    def trainable(config):
        _fence_gpu_memory(resources_per_trial)
        config = _resolve_choices(config, choice_options)
        reported = []

        def report(step, value):
            # A non-finite running value means no labelled rows yet, not a bad candidate —
            # reporting it would let the scheduler judge a rung that measured nothing.
            if not _finite(value):
                return
            reported.append(step)
            tune.report({metric: value, _STEP: step})

        try:
            value = trial_fn(config, report)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            # Letting the OOM escape reports the trial to Optuna as FAIL, and TPE draws only
            # on COMPLETE and PRUNED — so the sampler learns nothing and keeps proposing a
            # corner that cannot fit. NaN ends the trial without a usable score instead:
            # `_scored_value` reads it back as None, so it never reaches a ranking. None is
            # not an option in its place, since Ray hands the metric straight to
            # `optuna.Trial.report`, which requires a float.
            log.warning(f"Trial out of GPU memory, ending it unscored: {exc}")
            tune.report({metric: float("nan"), _STEP: last_step})
            return
        # An objective that reported its own steps has already filed its last one; reporting
        # again would land a second result on the same step.
        if not reported:
            tune.report({metric: value, _STEP: last_step})

    trainable_res = tune.with_resources(trainable, resources_per_trial) if resources_per_trial else trainable
    tuner = tune.Tuner(
        trainable_res,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=n_trials,
            max_concurrent_trials=max_parallel,
            search_alg=OptunaSearch(metric=metric, mode=mode, seed=seed, points_to_evaluate=seeded or None),
            scheduler=(
                _laddered_scheduler(
                    max_steps=max_steps,
                    metric=metric,
                    mode=mode,
                    is_exempt=lambda config: _is_seeded(config, seeded),
                )
                if max_steps
                else None
            ),
        ),
    )
    results = tuner.fit()

    # Never rank an unscored trial: one that died carries neither a value nor a config, so
    # ranking it would publish `None` as the winning config. Nor a stopped one: its objective
    # covers fewer steps than a full trial's, so the two are not the same measurement.
    pool = [
        r
        for r in results
        if getattr(r, "error", None) is None and _scored_value(r, metric) is not None and _reached_full(r, max_steps)
    ]
    if not pool:
        # Mirror the Optuna path's actionable failure. Ray otherwise surfaces this as an
        # AttributeError deep in config resolution, on the box that costs the most to rent.
        errored = sum(1 for r in results if r.error is not None)
        laddered = f", none reaching step {max_steps}" if max_steps else ""
        raise RuntimeError(
            f"HPO search produced no usable trial ({len(results)} trials, {errored} errored"
            f"{laddered}). If trials errored, check the logs above for the first traceback — "
            "an OOM or a NaN objective (e.g. an unlabeled target column) is the usual cause."
        )

    def _objective(r):
        # Missing/None metric sorts to the worst end regardless of mode — a signed default
        # (e.g. -inf in max mode) would otherwise make an unscored trial win.
        v = _scored_value(r, metric)
        if v is None:
            return float("inf")
        return v if mode == "min" else -v

    best = min(pool, key=_objective)
    trials = _resolve_trial_records(results, metric=metric, choice_options=choice_options, max_steps=max_steps)
    return HpoResult(
        best_config=_resolve_choices(best.config, choice_options),
        best_value=_scored_value(best, metric),
        metric=metric,
        mode=mode,
        n_trials=len(trials),
        trials=trials,
    )


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
