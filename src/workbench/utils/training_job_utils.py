"""Utilities for reading back what a model's SageMaker training job did.

These read from outside the job — CloudWatch metrics and the artifacts the job left in S3.
The code that runs *inside* a training container lives in ``workbench.training``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

# Workbench Imports
from workbench.utils.model_utils import extracted_artifact, model_instance_info

# Set up the log
log = logging.getLogger("workbench")


def get_hpo_results(workbench_model: Any) -> Optional[dict]:
    """Get the hyperparameter-search results for a Workbench model.

    HPO writes its audit trail to the training job's ``output.tar.gz``, a sibling of the
    ``model.tar.gz`` that ``model_data_url()`` points at. A None return means the model was
    not hyperparameter-searched, so this doubles as the "was this an HPO model?" check.

    Reading the returned values: ``best_value`` and ``baseline_value`` share the re-rank's
    basis, so their difference is the margin the publish decision turned on.
    ``search_best_value`` comes from the search phase and is not comparable to them when
    ``rerank_fresh_split`` is true — the re-rank scored on a different fold partition.

    Args:
        workbench_model: Workbench model object

    Returns:
        dict: ``best_config.json`` contents plus ``rerank``/``trials`` DataFrames, or None
        when the model has no search artifacts.
    """
    model_artifact_uri = workbench_model.model_data_url()
    if model_artifact_uri is None:
        log.warning(f"No model artifact found for {workbench_model.name}")
        return None

    output_uri = model_artifact_uri.rsplit("/", 1)[0] + "/output.tar.gz"
    with extracted_artifact(output_uri) as artifact_dir:
        if artifact_dir is None:
            return None
        best_config_path = os.path.join(artifact_dir, "best_config.json")
        if not os.path.exists(best_config_path):
            return None
        with open(best_config_path, "r") as f:
            results = json.load(f)
        for key, filename in (("rerank", "hpo_rerank.csv"), ("trials", "hpo_trials.csv")):
            csv_path = os.path.join(artifact_dir, filename)
            results[key] = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    return results


def get_hpo_search_space(workbench_model: Any) -> Optional[pd.DataFrame]:
    """Get the HPO search space for a Workbench model's framework.

    Describes what a search *would* explore, and where each knob sits when nobody tunes
    it — inspectable without running anything. This is the framework's full default space,
    not the space a particular search used; for that, read the ``hyperparameters`` column
    of :func:`get_hpo_results`' ``trials`` frame. To search a subset, set
    ``hpo["search_space"]`` at training time.

    Args:
        workbench_model: Workbench model object

    Returns:
        pd.DataFrame: one row per knob with three pinned columns — ``knob``, ``default``,
        ``dist`` — plus ``spec``, a JSON object carrying whatever fields that ``dist`` has
        (``low``/``high``/``step``/``log`` for a range, ``options`` for a categorical).
        ``json.loads`` the cell to read it. None when the model's framework has no HPO
        support.
    """
    # getattr on the model too: a model that doesn't resolve never gets a framework set.
    framework = getattr(workbench_model, "model_framework", None)
    framework = getattr(framework, "value", framework)

    # Deferred: workbench.training is training-only by contract. The search-space modules
    # are importable without the training extras (they defer optuna/ray/framework imports),
    # so this works from a lean environment such as the dashboard.
    from workbench.training.hpo_harness import SEARCH_SPACE_MODULES, SearchSpace

    if framework not in SEARCH_SPACE_MODULES:
        log.warning(f"No HPO search space for framework '{framework}' ({', '.join(SEARCH_SPACE_MODULES)} have one)")
        return None
    return SearchSpace(framework).to_frame()


# A search under this many scored trials cannot support an importance estimate.
_MIN_TRIALS_FOR_IMPORTANCE = 10

# How far above a random column's importance the top knob must sit to be worth reading.
# A knob merely *beating* noise is not evidence — the two estimates carry comparable
# uncertainty at these trial counts. Calibrated on 6-knob searches at 24 trials: a knob that
# really drives the objective clears the floor by 14x or more, while pure noise sits at ~1.4x
# (p90 2.3x). At 3x the warning catches 93% of no-signal searches and never fires on a real
# one — the populations are far enough apart that the exact cut hardly matters.
_NOISE_FLOOR_MARGIN = 3.0


def _encode_knobs(hyperparameters: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Encode each knob's trial values as floats for the surrogate, keeping the originals.

    Numeric knobs pass through. A categorical (``ffn_hidden_dim`` holds dash-string FFN
    shapes) is ordinal-coded by its sorted distinct values — an arbitrary but leak-free
    order, unlike ranking the categories by their objective.

    Returns:
        tuple[pd.DataFrame, dict]: the float-encoded frame, and ``{knob: (values, codes)}``
        pairing the values a user would actually set with their encoded positions.
    """
    knobs = pd.DataFrame([json.loads(h) for h in hyperparameters], index=hyperparameters.index)
    encoded, levels = pd.DataFrame(index=knobs.index), {}
    for knob in knobs.columns:
        numeric = pd.to_numeric(knobs[knob], errors="coerce")
        if numeric.notna().all():
            encoded[knob] = numeric.astype(float)
            values = sorted(numeric.unique().tolist())
            # An int knob reports an int back — a width of 7, not 7.0
            if pd.api.types.is_integer_dtype(knobs[knob]):
                values = [int(v) for v in values]
            levels[knob] = (values, [float(v) for v in values])
        else:
            # Sorting on the string form keeps mixed cells orderable; the original value is
            # carried alongside so a scalar width reports as 900, not "900".
            as_text = knobs[knob].astype(str)
            keys = sorted(as_text.unique().tolist())
            originals = {text: knobs[knob][as_text == text].iloc[0] for text in keys}
            encoded[knob] = as_text.map({text: i for i, text in enumerate(keys)}).astype(float)
            levels[knob] = ([originals[text] for text in keys], [float(i) for i in range(len(keys))])
    return encoded, levels


def _partial_dependence(model: Any, x: pd.DataFrame, knob: str, grid: list) -> "np.ndarray":
    """Mean surrogate prediction with ``knob`` pinned to each grid value, others left as-is.

    Marginalizes the other knobs out, which is what makes this readable as one knob's own
    effect — a search allocates trials adaptively, so raw per-value group means are
    confounded by whatever else the sampler was exploring at the time.
    """
    means = []
    for value in grid:
        pinned = x.copy()
        pinned[knob] = value
        means.append(float(model.predict(pinned).mean()))
    return np.array(means)


def _warn_if_below_noise_floor(x, y) -> None:
    """Log a warning when the top knob's importance is within reach of pure noise.

    Impurity importance is biased toward whatever offers the most split points, so a
    continuous knob outranks a 3-level one even when neither carries signal (Strobl et al.,
    BMC Bioinformatics 2007). Measure that directly: refit with one continuous junk column
    and compare the best real knob against it. Anything within reach of it is unmeasured,
    however large its normalized share looks.

    Both numbers come from *this* fit, never from the reported one. ``feature_importances_``
    is normalized within a fit, so a share read across N columns is not comparable to one
    read across N+1 — mixing them inflates the real knob purely for having fewer columns to
    divide among, always in the direction of staying silent.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    junk = "_noise_continuous"
    probe = x.copy()
    probe[junk] = np.random.default_rng(42).uniform(size=len(x))

    model = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=2).fit(probe, y)
    shares = dict(zip(probe.columns, model.feature_importances_))
    floor = float(shares.pop(junk))
    top_knob, top = max(shares.items(), key=lambda kv: kv[1], default=(None, 0.0))

    if top <= floor * _NOISE_FLOOR_MARGIN:
        log.warning(
            f"Hyperparameter importance is at the noise floor: {top_knob} scores {top:.3f} against "
            f"{floor:.3f} for a random column on the same {len(x)} trials. Rank the knobs by "
            "'effect' against your run-to-run spread instead — the shares are not separable here."
        )


def get_hpo_importance(workbench_model: Any) -> Optional[pd.DataFrame]:
    """Rank a searched model's knobs by how much they moved the objective.

    Fits a random-forest surrogate to the search's own trials, then reads two different
    things off it. ``importance`` is the surrogate's split-based importance, normalized
    across knobs — good for ranking, but it always sums to 1, so in a search where nothing
    mattered something still looks important. ``effect`` is the absolute read: how far the
    objective moves across that knob's range, as a percentage of the objective. **A knob is
    only worth tuning when both are high** — a large share of a negligible total is noise.

    These are observational estimates from an adaptive sampler, not a controlled ablation,
    and a typical search is a few dozen trials over several knobs. Treat the ordering as
    directional.

    Args:
        workbench_model: Workbench model object

    Returns:
        pd.DataFrame: one row per searched knob, most important first, with ``knob``,
        ``importance`` (shares summing to 1), ``effect`` (percent of the objective), and
        ``best`` (the knob's value where the objective is lowest, other knobs averaged
        out — meaningless when ``effect`` is small). None when the model was not searched.
    """
    results = get_hpo_results(workbench_model)
    if results is None or results.get("trials") is None:
        return None

    trials = results["trials"]
    if "kind" in trials:
        trials = trials[trials["kind"] == "trial"]
    # Pruned trials only ever scored a partial ensemble — fewer members than a full run, which
    # on a holdout objective reads systematically worse rather than merely noisier. Fitting the
    # surrogate to both is fitting it to a mixture of two objectives, and since a trial is
    # pruned precisely for looking bad early, the mixture lines up with the knobs being ranked.
    # The Ray backend records ``completed``; the Optuna backend records a ``state``.
    if "completed" in trials:
        trials = trials[trials["completed"].astype(bool)]
    elif "state" in trials:
        trials = trials[trials["state"] == "COMPLETE"]
    trials = trials.dropna(subset=["value"])
    if len(trials) < _MIN_TRIALS_FOR_IMPORTANCE:
        log.warning(f"Only {len(trials)} scored trials — too few to estimate hyperparameter importance")
        return None

    x, levels = _encode_knobs(trials["hyperparameters"])
    y = trials["value"].astype(float)

    from sklearn.ensemble import RandomForestRegressor

    surrogate = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=2).fit(x, y)
    _warn_if_below_noise_floor(x, y)

    scale = abs(float(y.mean())) or 1.0
    rows = []
    for knob, importance in zip(x.columns, surrogate.feature_importances_):
        values, codes = levels[knob]
        if len(values) < 2:  # never varied, so the search says nothing about it
            rows.append({"knob": knob, "importance": 0.0, "effect": 0.0, "best": values[0]})
            continue
        curve = _partial_dependence(surrogate, x, knob, codes)
        rows.append(
            {
                "knob": knob,
                "importance": float(importance),
                "effect": 100 * float(curve.max() - curve.min()) / scale,
                "best": values[int(curve.argmin())],
            }
        )

    # `best` holds each knob's native type (an int width, a float rate, a shape string), so
    # the column is built as object rather than letting pandas upcast the mix to float.
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return pd.DataFrame(
        {
            "knob": [row["knob"] for row in rows],
            "importance": [row["importance"] for row in rows],
            "effect": [row["effect"] for row in rows],
            "best": pd.Series([row["best"] for row in rows], dtype=object),
        }
    )


_UTILIZATION_METRICS = {
    "cpu": "CPUUtilization",
    "memory": "MemoryUtilization",
    "gpu": "GPUUtilization",
    "gpu_memory": "GPUMemoryUtilization",
    "disk": "DiskUtilization",
}

# The metrics CloudWatch sums across devices, and the per-device column derived from each
_PER_DEVICE_COLUMNS = {"cpu": "cpu_per_core", "gpu": "gpu_per_device"}


def get_training_utilization_details(workbench_model: Any) -> Optional[pd.DataFrame]:
    """Per-minute utilization for a model's training job, as SageMaker publishes it.

    ``cpu`` and ``gpu`` are summed across devices, so a busy 16-core box reads 1600;
    ``cpu_per_core`` and ``gpu_per_device`` are the per-device reads. An HPO run is one
    training job, so this covers the whole search rather than any single trial.

    Args:
        workbench_model: Workbench model object

    Returns:
        pd.DataFrame: one row per minute, indexed by UTC timestamp, with ``attrs`` carrying
            the job name, instance type/count, and device counts. None for a model copy, or
            a job past CloudWatch's 15-day retention of 1-minute data.
    """
    job_name = workbench_model.training_job_name
    if job_name is None:
        log.warning(f"No training job for {workbench_model.name} (a model copy has none)")
        return None

    session = workbench_model.boto3_session
    job = session.client("sagemaker").describe_training_job(TrainingJobName=job_name)
    resources = job["ResourceConfig"]
    instance_type, instance_count = resources["InstanceType"], resources["InstanceCount"]

    # The job's own window, so the query spans exactly the training run
    start, end = job["TrainingStartTime"], job.get("TrainingEndTime", datetime.now(timezone.utc))

    # One dimension per instance: SageMaker names them algo-1, algo-2, ... and each reports
    # its own utilization, so a distributed job needs a query per host.
    queries = [
        {
            "Id": f"m{i}_{key}",
            "Label": f"{key}|algo-{i + 1}",
            "MetricStat": {
                "Metric": {
                    "Namespace": "/aws/sagemaker/TrainingJobs",
                    "MetricName": metric_name,
                    "Dimensions": [{"Name": "Host", "Value": f"{job_name}/algo-{i + 1}"}],
                },
                "Period": 60,
                "Stat": "Average",
            },
        }
        for i in range(instance_count)
        for key, metric_name in _UTILIZATION_METRICS.items()
    ]

    cloudwatch = session.client("cloudwatch")
    results = []
    paginator = cloudwatch.get_paginator("get_metric_data")
    for page in paginator.paginate(MetricDataQueries=queries, StartTime=start, EndTime=end):
        results.extend(page["MetricDataResults"])

    # Metrics the instance does not report (GPU on a CPU box) come back empty and are dropped
    series = {}
    for result in results:
        if not result["Timestamps"]:
            continue
        label = result["Label"]
        column = label.split("|")[0] if instance_count == 1 else label.replace("|", "_")
        series[column] = pd.Series(result["Values"], index=pd.to_datetime(result["Timestamps"]))

    if not series:
        log.warning(f"No utilization datapoints for {job_name} (past CloudWatch's 15-day retention?)")
        return None

    df = pd.DataFrame(series).sort_index()
    df.index.name = "timestamp"

    # Per-device reads, when the instance's hardware counts are known. Only cpu and gpu are
    # summed across devices; gpu_memory and the rest are already true percentages.
    info = model_instance_info()
    match = info[info["Instance Name"] == instance_type]
    num_cpus = int(match["Num CPUs"].iloc[0]) if not match.empty else None
    num_gpus = int(match["Num GPUs"].iloc[0]) if not match.empty else None

    counts = {"cpu": num_cpus, "gpu": num_gpus}
    for base, derived in _PER_DEVICE_COLUMNS.items():
        if not counts[base]:
            continue
        for column in [c for c in df.columns if c == base or c.startswith(f"{base}_algo-")]:
            df[column.replace(base, derived, 1)] = df[column] / counts[base]

    # Each per-device column sits beside the metric it normalizes
    ordered = []
    for base in _UTILIZATION_METRICS:
        for name in (base, _PER_DEVICE_COLUMNS.get(base)):
            if name:
                ordered.extend(c for c in df.columns if c == name or c.startswith(f"{name}_algo-"))
    df = df[ordered + [c for c in df.columns if c not in ordered]]

    df.attrs.update(
        training_job_name=job_name,
        instance_type=instance_type,
        instance_count=instance_count,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )
    return df


def get_training_utilization(workbench_model: Any) -> Optional[pd.DataFrame]:
    """Was the training instance the right pick? One row per metric, over the whole job.

    A ``gpu_per_device`` median above ~70% means GPU-bound and sized about right; below ~30%
    means the GPU sat idle. High ``cpu_per_core`` beside low ``gpu_per_device`` points at
    featurization rather than the model. Median × device count is the sustained load in whole
    devices — 4 GPUs at a 60% median is 2.4 GPUs of work. ``peak`` is a 1-minute average, too
    coarse to rule out a brief burst.

    Args:
        workbench_model: Workbench model object

    Returns:
        pd.DataFrame: one row per metric with ``mean``, ``median``, and ``peak``, the hardware
            on the index name. None when :func:`get_training_utilization_details` finds
            nothing to summarize.
    """
    df = get_training_utilization_details(workbench_model)
    if df is None:
        return None

    summary = pd.DataFrame({"mean": df.mean(), "median": df.median(), "peak": df.max()}).round(1)

    # The hardware rides on the index name so it shows up in the frame's own repr
    attrs = df.attrs
    hardware = [attrs["instance_type"]]
    counts = [f"{attrs[key]} {label}" for key, label in (("num_cpus", "CPUs"), ("num_gpus", "GPUs")) if attrs[key]]
    if counts:
        hardware.append(f"({', '.join(counts)})")
    if attrs["instance_count"] > 1:
        hardware.append(f"x{attrs['instance_count']} instances")
    summary.index.name = " ".join(hardware)

    summary.attrs.update(attrs)
    return summary
