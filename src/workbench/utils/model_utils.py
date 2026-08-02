"""Model Utilities for Workbench models"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import importlib.resources
from contextlib import contextmanager
from pathlib import Path
import os
import json
import tempfile
import tarfile
import awswrangler as wr
from typing import Iterator, Optional, Dict, Any, TYPE_CHECKING
from scipy.stats import norm

if TYPE_CHECKING:
    from workbench.api import Model
    from workbench.algorithms.models.noise_model import NoiseModel
    from workbench.algorithms.models.cleanlab_model import CleanlabModels

# Set up the log
log = logging.getLogger("workbench")


def model_instance_info() -> pd.DataFrame:
    """Instance reference for the Model: hardware, us-east-1 on-demand price, and role.

    Price per Hour is the rate for the instance's Usage — SageMaker charges a different
    rate for the same instance hosting an endpoint versus running a training job. The
    Training rows are the ladders in
    ``workbench.core.transforms.features_to_model.INSTANCE_LADDERS``.
    """
    data = [
        {
            "Instance Name": "ml.t2.medium",
            "vCPUs": 2,
            "Memory": 4,
            "Price per Hour": 0.06,
            "Category": "General",
            "Architecture": "x86_64",
            "Usage": "Hosting",
        },
        {
            "Instance Name": "ml.m7i.large",
            "vCPUs": 2,
            "Memory": 8,
            "Price per Hour": 0.12,
            "Category": "General",
            "Architecture": "x86_64",
            "Usage": "Hosting",
        },
        {
            "Instance Name": "ml.c7i.large",
            "vCPUs": 2,
            "Memory": 4,
            "Price per Hour": 0.11,
            "Category": "Compute",
            "Architecture": "x86_64",
            "Usage": "Hosting",
        },
        {
            "Instance Name": "ml.c7i.xlarge",
            "vCPUs": 4,
            "Memory": 8,
            "Price per Hour": 0.21,
            "Category": "Compute",
            "Architecture": "x86_64",
            "Usage": "Hosting",
        },
        {
            "Instance Name": "ml.c7g.large",
            "vCPUs": 2,
            "Memory": 4,
            "Price per Hour": 0.09,
            "Category": "Compute",
            "Architecture": "arm64",
            "Usage": "Hosting",
        },
        {
            "Instance Name": "ml.c7g.xlarge",
            "vCPUs": 4,
            "Memory": 8,
            "Price per Hour": 0.17,
            "Category": "Compute",
            "Architecture": "arm64",
            "Usage": "Hosting",
        },
        {
            "Instance Name": "ml.m5.xlarge",
            "vCPUs": 4,
            "Memory": 16,
            "Price per Hour": 0.23,
            "Category": "General",
            "Architecture": "x86_64",
            "Usage": "Training",
        },
        {
            "Instance Name": "ml.c7i.4xlarge",
            "vCPUs": 16,
            "Memory": 32,
            "Price per Hour": 0.86,
            "Category": "Compute",
            "Architecture": "x86_64",
            "Usage": "Training",
        },
        {
            "Instance Name": "ml.g6.2xlarge",
            "vCPUs": 8,
            "Memory": 32,
            "Price per Hour": 1.21,
            "Category": "GPU",  # 1x NVIDIA L4 24GB
            "Architecture": "x86_64",
            "Usage": "Training",
        },
        {
            "Instance Name": "ml.g6.12xlarge",
            "vCPUs": 48,
            "Memory": 192,
            "Price per Hour": 5.61,
            "Category": "GPU",  # 4x NVIDIA L4 24GB
            "Architecture": "x86_64",
            "Usage": "Training",
        },
        {
            "Instance Name": "ml.g5.12xlarge",
            "vCPUs": 48,
            "Memory": 192,
            "Price per Hour": 7.09,
            "Category": "GPU",  # 4x NVIDIA A10G 24GB
            "Architecture": "x86_64",
            "Usage": "Training",
        },
    ]
    return pd.DataFrame(data)


def instance_architecture(instance_name: str) -> str:
    """Get the architecture for the given instance name"""
    info = model_instance_info()
    return info[info["Instance Name"] == instance_name]["Architecture"].values[0]


def supported_instance_types(arch: str = "x86_64", usage: str = "Hosting") -> list:
    """Get the supported instance types for the given architecture and usage (Hosting/Training)"""

    info = model_instance_info()
    matches = (info["Architecture"] == arch) & (info["Usage"] == usage)
    return info[matches]["Instance Name"].tolist()


def get_custom_script_path(package: str, script_name: str) -> Path:
    package_path = importlib.resources.files(f"workbench.model_scripts.custom_models.{package}")
    script_path = package_path / script_name
    return script_path


def copy_model_artifacts(model: "Model", dst_name: str) -> str:
    """Stage a model copy's S3 artifacts under the destination's training dir.

    Copies the frozen model.tar.gz and its sibling output.tar.gz (the training job's
    output channel, which carries the HPO audit trail) plus the top-level
    training-capture files (validation_predictions.csv, shap_*) into
    {models_s3_path}/{dst_name}/training/. The frozen artifact lives in the copy's own
    dir so it's immune to the source's delete-then-create churn.

    Args:
        model (Model): The source model being copied
        dst_name (str): Name of the destination model group

    Returns:
        str: The frozen model.tar.gz S3 URL for the copy's container spec
    """
    src_url = model.model_data_url()
    dst_training_path = f"{model.models_s3_path}/{dst_name}/training"
    session = model.boto3_session

    # Freeze the artifact under the copy's own training dir, keeping output.tar.gz beside
    # it so readers that resolve it from model_data_url() (get_hpo_results) work on the copy
    src_dir = src_url.rsplit("/", 1)[0]
    output_url = f"{src_dir}/output.tar.gz"
    src_objs = [src_url]
    if wr.s3.does_object_exist(output_url, boto3_session=session):
        src_objs.append(output_url)
    wr.s3.copy_objects(
        src_objs,
        source_path=src_dir,
        target_path=dst_training_path,
        boto3_session=session,
    )

    # Carry the top-level training-capture files (skip timestamped job-output subdirs)
    prefix = model.model_training_path + "/"
    training_objs = [o for o in wr.s3.list_objects(path=prefix) if "/" not in o[len(prefix) :]]
    if training_objs:
        wr.s3.copy_objects(
            training_objs,
            source_path=model.model_training_path,
            target_path=dst_training_path,
            boto3_session=session,
        )

    return f"{dst_training_path}/model.tar.gz"


_VALID_UQ_VERSIONS = ("v0", "v1", "v2")


def _resolve_uq_version(model: "Model", version: Optional[str]) -> str:
    """Resolve the effective UQ version for a model artifact.

    Order of precedence:
        1. Explicit `version` argument ("v0", "v1", or "v2").
        2. `hyperparameters["uq_version"]` from the model artifact.
        3. Default "v0".
    """
    if version is not None:
        return version

    # Try to read uq_version from the model's hyperparameters
    try:
        hp = model.hyperparameters() if hasattr(model, "hyperparameters") else None
        if hp and "uq_version" in hp:
            return str(hp["uq_version"])
    except Exception:  # noqa: BLE001 — best-effort lookup, fall through to default
        pass

    return "v0"


def uq_model_local(
    model: Model,
    version: Optional[str] = None,
    refresh_proximity: bool = False,
    radius: int = 2,
    n_bits: int = 4096,
) -> "UQModelV0 | UQModelV1 | UQModelV2":  # noqa: F821
    """Load the fitted UQModel (V0, V1, or V2) from this Model's artifact.

    Pairs with the existing `fp_prox_model()` / `proximity_model()` factory pattern:
        model = Model("my-model")
        rm = model.uq_model()
        out = rm.predict(test_df[["smiles"]], predictions, prediction_std)

    Args:
        model: The Workbench Model whose artifact contains a fitted UQModel.
        version: Which UQ version to load — ``"v0"`` (isotonic on prediction+std),
            ``"v1"`` (proximity-augmented RF), or ``"v2"`` (pure applicability-domain
            from fingerprint neighbors). If ``None``, reads
            ``hyperparameters["uq_version"]`` from the bundle and falls back
            to ``"v0"``.
        refresh_proximity: V1/V2 only. If False (default), use the proximity backend
            that was embedded in the model artifact at training time — exact
            reference set used to fit the residual estimator, reproducible, no
            fingerprint recomputation. If True, build a fresh FingerprintProximity
            from the current source FeatureSet. Ignored for V0 (no proximity).
        radius: Morgan fingerprint radius (only used for V1/V2 when refresh_proximity=True).
        n_bits: Morgan fingerprint bit width (only used for V1/V2 when refresh_proximity=True).

    Returns:
        A ready-to-use UQModelV0, UQModelV1, or UQModelV2 instance.

    Raises:
        FileNotFoundError: If the requested version's artifact is not in the bundle.
    """
    from workbench.algorithms.dataframe.uq_model_v0 import UQModelV0  # noqa: F401
    from workbench.algorithms.dataframe.uq_model_v1 import UQModelV1  # noqa: F401
    from workbench.algorithms.dataframe.uq_model_v2 import UQModelV2  # noqa: F401

    model_artifact_uri = model.model_data_url()
    if model_artifact_uri is None:
        raise ValueError(f"No model artifact found for {model.name}")

    effective_version = _resolve_uq_version(model, version)
    if effective_version not in _VALID_UQ_VERSIONS:
        raise ValueError(f"Unknown UQ version '{effective_version}' (expected one of {_VALID_UQ_VERSIONS})")

    # V1/V2 share the proximity artifact; optionally build a fresh one to override
    fresh_prox = None
    if effective_version in ("v1", "v2") and refresh_proximity:
        from workbench.utils.prox_utils import fingerprint_prox_model_local

        fresh_prox = fingerprint_prox_model_local(model, radius=radius, n_bits=n_bits)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_tar_path = os.path.join(tmpdir, "model.tar.gz")
        wr.s3.download(path=model_artifact_uri, local_file=local_tar_path)
        safe_extract_tarfile(local_tar_path, tmpdir)

        if effective_version == "v0":
            return UQModelV0.load(tmpdir)

        if effective_version == "v1":
            if not os.path.exists(os.path.join(tmpdir, "uq_model.joblib")):
                raise FileNotFoundError(
                    f"Model '{model.name}' does not have a fitted UQModelV1 "
                    "(expected uq_model.joblib in the model artifact)."
                )
            return UQModelV1.load(tmpdir, prox=fresh_prox)

        # v2
        if not os.path.exists(os.path.join(tmpdir, UQModelV2.METADATA_FILENAME)):
            raise FileNotFoundError(
                f"Model '{model.name}' does not have a fitted UQModelV2 "
                f"(expected {UQModelV2.METADATA_FILENAME} in the model artifact)."
            )
        return UQModelV2.load(tmpdir, prox=fresh_prox)


def noise_model_local(model: Model) -> NoiseModel:
    """Create a NoiseModel for detecting noisy/problematic samples in a Model's training data.

    Args:
        model (Model): The Model used to create the noise model

    Returns:
        NoiseModel: The noise model with precomputed noise scores for all samples
    """
    from workbench.algorithms.models.noise_model import NoiseModel  # noqa: F401 (avoid circular import)
    from workbench.api import Model, FeatureSet  # noqa: F401 (avoid circular import)

    # Get Feature and Target Columns from the existing given Model
    features = model.features()
    target = model.target()

    # Backtrack our FeatureSet to get the ID column
    fs = FeatureSet(model.get_input())
    id_column = fs.id_column

    # Create the NoiseModel from both the full FeatureSet and the Model training data
    full_df = fs.pull_dataframe()
    model_df = model.training_view().pull_dataframe()

    # Mark rows that are in the model
    model_ids = set(model_df[id_column])
    full_df["in_model"] = full_df[id_column].isin(model_ids)

    # Create and return the NoiseModel
    return NoiseModel(full_df, id_column, features, target)


def cleanlab_model_local(model: Model) -> CleanlabModels:
    """Create a CleanlabModels instance for detecting data quality issues in a Model's training data.

    Args:
        model (Model): The Model used to create the cleanlab models

    Returns:
        CleanlabModels: Label-quality analysis with helpers like label_issues(),
            epistemic_uncertainty(), and the native clean_learning()/datalab() objects.
    """
    from workbench.algorithms.models.cleanlab_model import CleanlabModels  # noqa: F401 (avoid circular import)
    from workbench.api import Model, FeatureSet  # noqa: F401 (avoid circular import)

    # Get Feature and Target Columns from the existing given Model
    features = model.features()
    target = model.target()
    model_type = model.model_type

    # Backtrack our FeatureSet to get the ID column
    fs = FeatureSet(model.get_input())
    id_column = fs.id_column

    # Get the full FeatureSet data
    full_df = fs.pull_dataframe()

    # Create and return the CleanlabModels instance
    return CleanlabModels(full_df, id_column, features, target, model_type=model_type)


def safe_extract_tarfile(tar_path: str, extract_path: str) -> None:
    """
    Extract a tarball safely, using data filter if available.

    The filter parameter was backported to Python 3.8+, 3.9+, 3.10.13+, 3.11+
    as a security patch, but may not be present in older patch versions.
    """
    with tarfile.open(tar_path, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(path=extract_path, filter="data")
        else:
            tar.extractall(path=extract_path)


@contextmanager
def extracted_artifact(artifact_uri: str) -> Iterator[Optional[str]]:
    """Download an S3 tarball and yield the temp directory it extracted into.

    Yields None when the object can't be fetched — callers name a specific artifact and a
    bundle need not contain it (only searched models write ``output.tar.gz``). The directory
    is removed on exit, so read what you need inside the ``with``.

    Args:
        artifact_uri (str): S3 URI of a .tar.gz artifact.

    Yields:
        str | None: Path to the extracted directory, or None if the download failed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_tar_path = os.path.join(tmpdir, "artifact.tar.gz")
        try:
            wr.s3.download(path=artifact_uri, local_file=local_tar_path)
        except Exception as e:
            log.debug(f"Could not download artifact {artifact_uri}: {e}")
            yield None
            return
        safe_extract_tarfile(local_tar_path, tmpdir)
        yield tmpdir


def _load_json_from_artifact(artifact_uri: str, filename: str) -> Optional[dict]:
    """Read one JSON file from the base directory of an S3 tarball artifact.

    Args:
        artifact_uri (str): S3 URI of the .tar.gz artifact.
        filename (str): File to read from the archive's base directory.

    Returns:
        dict: The parsed JSON, or None if the artifact or file is absent/unreadable.
    """
    with extracted_artifact(artifact_uri) as artifact_dir:
        if artifact_dir is None:
            return None
        path = os.path.join(artifact_dir, filename)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            log.debug(f"Loaded {filename} from {artifact_uri}")
            return data
        except Exception as e:
            log.warning(f"Failed to load {filename} from {artifact_uri}: {e}")
            return None


def load_category_mappings_from_s3(model_artifact_uri: str) -> Optional[dict]:
    """
    Download and extract category mappings from a model artifact in S3.

    Args:
        model_artifact_uri (str): S3 URI of the model artifact.

    Returns:
        dict: The loaded category mappings or None if not found.
    """
    return _load_json_from_artifact(model_artifact_uri, "category_mappings.json")


def load_hyperparameters_from_s3(model_artifact_uri: str) -> Optional[dict]:
    """
    Download and extract hyperparameters from a model artifact in S3.

    Args:
        model_artifact_uri (str): S3 URI of the model artifact (model.tar.gz).

    Returns:
        dict: The loaded hyperparameters or None if not found.
    """
    return _load_json_from_artifact(model_artifact_uri, "hyperparameters.json")


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

    Numeric knobs pass through. A categorical (``ffn_hidden_dim`` mixes int widths with
    dash-string shapes) is ordinal-coded by its sorted distinct values — an arbitrary but
    leak-free order, unlike ranking the categories by their objective.

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


def get_model_hyperparameters(workbench_model: Any) -> Optional[dict]:
    """Get the hyperparameters used to train a Workbench model.

    Reads from Workbench meta (a cheap tag read). Models predating meta storage
    fall back to the model artifact and are backfilled into meta on first read.

    Args:
        workbench_model: Workbench model object

    Returns:
        dict: The hyperparameters used during training, or None if not found
    """
    # Fast path: hyperparameters cached in Workbench meta (a cheap tag read)
    hyperparameters = workbench_model.workbench_meta().get("workbench_hyperparameters")
    if hyperparameters is not None:
        return hyperparameters

    # Legacy fallback: pull from the model artifact (downloads + extracts model.tar.gz)
    model_artifact_uri = workbench_model.model_data_url()
    if model_artifact_uri is None:
        log.warning(f"No model artifact found for {workbench_model.name}")
        return None

    hyperparameters = load_hyperparameters_from_s3(model_artifact_uri)

    # Backfill meta so subsequent reads take the fast path
    if hyperparameters is not None:
        workbench_model.upsert_workbench_meta({"workbench_hyperparameters": hyperparameters})

    return hyperparameters


def uq_metrics(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Evaluate uncertainty quantification model with essential metrics.
    Args:
        df: DataFrame with predictions and uncertainty estimates. Must contain the target
            column, a "prediction" column, and a "prediction_std" column (required for
            CRPS and median_std). Quantile columns ("q_025", "q_975", "q_05", "q_95",
            "q_10", "q_90", "q_25", "q_75") are used for coverage/width when present;
            otherwise Gaussian bounds are derived from "prediction_std".
        target_col: Name of the true target column in the DataFrame.
    Returns:
        Dictionary of computed metrics.
    """
    # Input Validation
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    if "prediction" not in df.columns:
        raise ValueError("Prediction column 'prediction' not found in DataFrame.")

    # Drop rows with NaN in any column the metrics depend on. UQ versions that
    # emit NaN for unscored compounds (no proximity match, etc.) would
    # otherwise poison np.median and scipy.spearmanr, which propagate NaN to
    # the entire scalar metric.
    n_total = len(df)
    candidate_cols = [
        "prediction",
        "prediction_std",
        "confidence",
        target_col,
        "q_025",
        "q_05",
        "q_10",
        "q_16",
        "q_25",
        "q_75",
        "q_84",
        "q_90",
        "q_95",
        "q_975",
    ]
    required_cols = [c for c in candidate_cols if c in df.columns]
    df = df.dropna(subset=required_cols)
    n_valid = len(df)
    if n_valid < n_total:
        log.info(f"UQ metrics: dropped {n_total - n_valid} of {n_total} rows with NaN in metric inputs")
    if n_valid == 0:
        log.warning("UQ metrics: no valid rows after dropping NaNs. Returning empty metrics.")
        return {}

    # --- Coverage and Interval Width ---
    if "q_025" in df.columns and "q_975" in df.columns:
        lower_95, upper_95 = df["q_025"], df["q_975"]
        lower_90, upper_90 = df["q_05"], df["q_95"]
        lower_80, upper_80 = df["q_10"], df["q_90"]
        lower_68 = df.get("q_16", df["q_10"])  # fallback to 80% interval
        upper_68 = df.get("q_84", df["q_90"])  # fallback to 80% interval
        lower_50, upper_50 = df["q_25"], df["q_75"]
    elif "prediction_std" in df.columns:
        lower_95 = df["prediction"] - 1.96 * df["prediction_std"]
        upper_95 = df["prediction"] + 1.96 * df["prediction_std"]
        lower_90 = df["prediction"] - 1.645 * df["prediction_std"]
        upper_90 = df["prediction"] + 1.645 * df["prediction_std"]
        lower_80 = df["prediction"] - 1.282 * df["prediction_std"]
        upper_80 = df["prediction"] + 1.282 * df["prediction_std"]
        lower_68 = df["prediction"] - 1.0 * df["prediction_std"]
        upper_68 = df["prediction"] + 1.0 * df["prediction_std"]
        lower_50 = df["prediction"] - 0.674 * df["prediction_std"]
        upper_50 = df["prediction"] + 0.674 * df["prediction_std"]
    else:
        raise ValueError(
            "Either quantile columns (q_025, q_975, q_25, q_75) or 'prediction_std' column must be present."
        )
    median_std = df["prediction_std"].median()
    coverage_95 = np.mean((df[target_col] >= lower_95) & (df[target_col] <= upper_95))
    coverage_90 = np.mean((df[target_col] >= lower_90) & (df[target_col] <= upper_90))
    coverage_80 = np.mean((df[target_col] >= lower_80) & (df[target_col] <= upper_80))
    coverage_68 = np.mean((df[target_col] >= lower_68) & (df[target_col] <= upper_68))
    median_width_95 = np.median(upper_95 - lower_95)
    median_width_90 = np.median(upper_90 - lower_90)
    median_width_80 = np.median(upper_80 - lower_80)
    median_width_50 = np.median(upper_50 - lower_50)
    median_width_68 = np.median(upper_68 - lower_68)

    # --- CRPS (measures calibration + sharpness) ---
    z = (df[target_col] - df["prediction"]) / df["prediction_std"]
    crps = df["prediction_std"] * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    mean_crps = np.mean(crps)

    # --- Interval Score @ 95% (penalizes miscoverage) ---
    alpha_95 = 0.05
    is_95 = (
        (upper_95 - lower_95)
        + (2 / alpha_95) * (lower_95 - df[target_col]) * (df[target_col] < lower_95)
        + (2 / alpha_95) * (df[target_col] - upper_95) * (df[target_col] > upper_95)
    )
    mean_is_95 = np.mean(is_95)

    # --- Interval to Error Correlation ---
    abs_residuals = np.abs(df[target_col] - df["prediction"])
    width_68 = upper_68 - lower_68

    # Spearman correlation for robustness
    interval_to_error_corr = spearmanr(width_68, abs_residuals)[0]

    # --- Confidence to Error Correlation ---
    # If confidence column exists, compute correlation (should be negative: high confidence = low error)
    confidence_to_error_corr = None
    if "confidence" in df.columns:
        confidence_to_error_corr = spearmanr(df["confidence"], abs_residuals)[0]

    # Collect results
    results = {
        "coverage_68": coverage_68,
        "coverage_80": coverage_80,
        "coverage_90": coverage_90,
        "coverage_95": coverage_95,
        "median_std": median_std,
        "median_width_50": median_width_50,
        "median_width_68": median_width_68,
        "median_width_80": median_width_80,
        "median_width_90": median_width_90,
        "median_width_95": median_width_95,
        "interval_to_error_corr": interval_to_error_corr,
        "confidence_to_error_corr": confidence_to_error_corr,
        "n_samples": len(df),
    }

    print("\n=== UQ Metrics ===")
    print(f"Coverage @ 68%: {coverage_68:.3f} (target: 0.68)")
    print(f"Coverage @ 80%: {coverage_80:.3f} (target: 0.80)")
    print(f"Coverage @ 90%: {coverage_90:.3f} (target: 0.90)")
    print(f"Coverage @ 95%: {coverage_95:.3f} (target: 0.95)")
    print(f"Median Prediction StdDev: {median_std:.3f}")
    print(f"Median 50% Width: {median_width_50:.3f}")
    print(f"Median 68% Width: {median_width_68:.3f}")
    print(f"Median 80% Width: {median_width_80:.3f}")
    print(f"Median 90% Width: {median_width_90:.3f}")
    print(f"Median 95% Width: {median_width_95:.3f}")
    print(f"CRPS: {mean_crps:.3f} (lower is better)")
    print(f"Interval Score 95%: {mean_is_95:.3f} (lower is better)")
    print(f"Interval/Error Corr: {interval_to_error_corr:.3f} (higher is better, target: >0.5)")
    if confidence_to_error_corr is not None:
        print(f"Confidence/Error Corr: {confidence_to_error_corr:.3f} (lower is better, target: <-0.5)")
    print(f"Samples: {len(df)}")
    return results


if __name__ == "__main__":
    """Exercise the Model Utilities"""
    from workbench.api import Model  # noqa: F811

    # Get the instance information
    print(model_instance_info())

    # Get the supported instance types
    print(supported_instance_types())

    # Get the architecture for the given instance
    print(instance_architecture("ml.c7i.large"))
    print(instance_architecture("ml.c7g.large"))

    # Get the custom script path
    print(get_custom_script_path("uq_models", "ensemble_xgb.template"))

    # Test loading hyperparameters
    m = Model("aqsol-regression")
    hyperparams = get_model_hyperparameters(m)
    print(hyperparams)

    # Test the proximity model
    # prox_model = proximity_model(m, "aqsol-prox")
    # print(prox_model)#
