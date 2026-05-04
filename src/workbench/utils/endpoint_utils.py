"""Endpoint Utilities for Workbench endpoints.

Includes admin / lookup helpers for endpoint output columns — the columns
an endpoint emits during inference. For feature endpoints (e.g. SMILES →
descriptors) these are the computed feature columns; for predictor endpoints
they are the prediction / confidence / quantile columns.

Endpoint contract metadata is registered in ParameterStore under the convention:

    /workbench/endpoints/<endpoint_name>/output_columns
    /workbench/endpoints/<endpoint_name>/input_columns

Write side (admin, from a deploy script):

    from workbench.utils.endpoint_utils import register_output_columns
    register_output_columns(end)   # idempotent — re-registers if the output shape changed

Read side (downstream model-training scripts):

    from workbench.utils.endpoint_utils import get_output_columns
    cols = get_output_columns("smiles-to-2d-v1")
"""

from __future__ import annotations

import boto3
from botocore.exceptions import ClientError
import logging
from typing import List, Union, Optional
import pandas as pd

# Workbench Imports
from workbench.api import FeatureSet, Model, Endpoint, ParameterStore

# Set up the log
log = logging.getLogger("workbench")

ENDPOINT_PARAM_PREFIX = "/workbench/endpoints"

# Provenance / preprocessing-metadata columns that feature endpoints emit
# alongside real features. Excluded from the registered output columns by
# `register_output_columns`. Add a column here only if it's truly not a feature
# (e.g. a record of the original SMILES before canonicalization).
NON_FEATURE_COLUMNS = frozenset(
    {
        "orig_smiles",
        "salt",
        "undefined_chiral_centers",
    }
)


def internal_model_data_url(endpoint_config_name: str, session: boto3.Session) -> Optional[str]:
    """
    Retrieves the S3 URL of the model.tar.gz file associated with a SageMaker endpoint configuration.

    Args:
        endpoint_config_name (str): The name of the SageMaker endpoint configuration.
        session (boto3.Session): An active boto3 session.

    Returns:
        Optional[str]: S3 URL of the model.tar.gz file if found, otherwise None.
    """
    try:
        sagemaker_client = session.client("sagemaker")

        # Retrieve the Endpoint Config
        endpoint_config = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)

        # Extract Model Name from Production Variants
        production_variants = endpoint_config.get("ProductionVariants", [])
        if not production_variants:
            log.critical(f"No production variants found for endpoint config: {endpoint_config_name}")
            return None

        model_name = production_variants[0].get("ModelName")
        if not model_name:
            log.critical(f"No model name found in production variants for endpoint config: {endpoint_config_name}")
            return None

        # Retrieve Model Details
        model_details = sagemaker_client.describe_model(ModelName=model_name)
        containers = model_details.get("Containers")
        if containers:
            # Handle serverless or multi-container models
            model_package_name = containers[0].get("ModelPackageName")
            if model_package_name:
                log.info(f"Model package name found: {model_package_name}")

                # Describe the model package to get the ModelDataUrl
                model_package_details = sagemaker_client.describe_model_package(ModelPackageName=model_package_name)
                model_data_url = (
                    model_package_details.get("InferenceSpecification", {})
                    .get("Containers", [{}])[0]
                    .get("ModelDataUrl")
                )
                if model_data_url:
                    log.info(f"Model data URL from package: {model_data_url}")
                    return model_data_url

        # Handle standard models
        model_data_url = model_details.get("PrimaryContainer", {}).get("ModelDataUrl")
        if model_data_url:
            log.info(f"Model data URL found: {model_data_url}")
            return model_data_url

        log.critical(f"No model data or package details found for model: {model_name}")
        return None

    except Exception as e:
        log.critical(f"Error retrieving model data URL for endpoint config {endpoint_config_name}: {e}")
        return None


def get_training_data(end: Endpoint) -> pd.DataFrame:
    """Code to get the training data from the FeatureSet used to train the Model

    Args:
        end (Endpoint): Endpoint to backtrace: End -> Model -> FeatureSet (training data)

    Returns:
        pd.DataFrame: Dataframe with the features from the FeatureSet
    """
    # Grab the FeatureSet by backtracking from the Endpoint
    fs = backtrack_to_fs(end)

    # Sanity check that we have a FeatureSet
    if fs is None:
        log.error("No FeatureSet found for this endpoint. Returning empty dataframe.")
        return pd.DataFrame()

    # Get the training data
    table = fs.view("training").table
    train_df = fs.query(f'SELECT * FROM "{table}" where training = TRUE')
    return train_df


def get_evaluation_data(end: Endpoint) -> pd.DataFrame:
    """Code to get the evaluation data from the FeatureSet NOT used for training

    Args:
        end (Endpoint): Endpoint to backtrace: End -> Model -> FeatureSet (evaluation data)

    Returns:
        pd.DataFrame: The training data in a dataframe with the features from the FeatureSet
    """
    # Grab the FeatureSet by backtracking from the Endpoint
    fs = backtrack_to_fs(end)

    # Sanity check that we have a FeatureSet
    if fs is None:
        log.error("No FeatureSet found for this endpoint. Returning empty dataframe.")
        return pd.DataFrame()

    # Get the evaluation data
    table = fs.view("training").table
    eval_df = fs.query(f'SELECT * FROM "{table}" where training = FALSE')
    return eval_df


def backtrack_to_fs(end: Endpoint) -> Union[FeatureSet, None]:
    """Code to Backtrack to FeatureSet: End -> Model -> FeatureSet

    Returns:
        FeatureSet (Union[FeatureSet, None]): The FeatureSet object or None if not found
    """

    # Sanity Check that we have a model
    model = Model(end.get_input())
    if not model.exists():
        log.error("No model found for this endpoint. Returning None.")
        return None

    # Now get the FeatureSet and make sure it exists
    fs = FeatureSet(model.get_input())
    if not fs.exists():
        log.error("No FeatureSet found for this endpoint. Returning None.")
        return None

    # Return the FeatureSet
    return fs


def is_monitored(endpoint_name: str, sagemaker_client: boto3.Session.client) -> bool:
    """Is monitoring enabled for this Endpoint?

    Args:
        endpoint_name: The name of the SageMaker Endpoint.
        sagemaker_client: Boto3 SageMaker client instance.

    Returns:
        True if monitoring is enabled, False otherwise.
    """
    try:
        response = sagemaker_client.list_monitoring_schedules(EndpointName=endpoint_name)
        return bool(response.get("MonitoringScheduleSummaries", []))
    except ClientError:
        return False


def output_columns_key(endpoint_name: str) -> str:
    """Return the ParameterStore key under which an endpoint's output
    columns are registered.

    Example:
        >>> output_columns_key("smiles-to-2d-v1")
        '/workbench/endpoints/smiles-to-2d-v1/output_columns'
    """
    return f"{ENDPOINT_PARAM_PREFIX}/{endpoint_name}/output_columns"


def input_columns_key(endpoint_name: str) -> str:
    """Return the ParameterStore key under which an endpoint's input
    columns are registered.

    Example:
        >>> input_columns_key("smiles-to-2d-v1")
        '/workbench/endpoints/smiles-to-2d-v1/input_columns'
    """
    return f"{ENDPOINT_PARAM_PREFIX}/{endpoint_name}/input_columns"


def lookup_cached_columns(endpoint, key: str, register_fn, kind: str) -> List[str]:
    """Cache-with-freshness lookup for an endpoint's registered column list.

    Used by both :meth:`Endpoint.output_columns` and :meth:`Endpoint.input_columns`.

    Reads ``key`` from ParameterStore. If it's missing, or the parameter's
    ``LastModifiedDate`` is older than the endpoint's ``modified()`` time,
    invokes ``register_fn(endpoint)`` to re-derive and rewrite the cache.

    Args:
        endpoint: The Workbench Endpoint instance.
        key: ParameterStore key (e.g. from :func:`output_columns_key`).
        register_fn: Callable taking the endpoint, returning the fresh column
            list and writing it to ParameterStore (e.g.
            :func:`register_output_columns`, :func:`register_input_columns`).
        kind: Short label for log messages (e.g. ``"output columns"``).

    Returns:
        List of column names — fresh from cache, or just-rewritten by
        ``register_fn``.
    """
    ps = ParameterStore()
    cols = ps.get(key)

    if cols is None:
        endpoint.log.important(
            f"Endpoint[{endpoint.name}]: no {kind} registered yet — deriving and caching."
        )
        return register_fn(endpoint)

    param_modified = ps.last_modified(key)
    try:
        endpoint_modified = endpoint.modified()
    except Exception:
        endpoint_modified = None

    if param_modified is not None and endpoint_modified is not None and endpoint_modified > param_modified:
        endpoint.log.important(
            f"Endpoint[{endpoint.name}]: cached {kind} are stale "
            f"(endpoint modified {endpoint_modified} > param modified {param_modified}) — re-deriving."
        )
        return register_fn(endpoint)

    return cols


def get_input_columns(endpoint_name: str) -> Optional[List[str]]:
    """Look up the input columns registered for an endpoint.

    Args:
        endpoint_name: e.g. ``"smiles-to-2d-v1"`` or ``"abalone-regression"``.

    Returns:
        List of input column names, or ``None`` if the endpoint hasn't
        registered its input columns yet. Call :func:`register_input_columns`
        from the endpoint's deploy script (or rely on the lazy fallback in
        :meth:`workbench.api.endpoint.Endpoint.input_columns`) to populate.
    """
    return ParameterStore().get(input_columns_key(endpoint_name))


def register_input_columns(endpoint, input_cols: Optional[List[str]] = None) -> List[str]:
    """Register an endpoint's input columns in ParameterStore.

    Two modes:

    1. **Auto-discovery (default, ``input_cols=None``):** backtraces
       endpoint → model and reads ``model.features()`` — the columns the
       endpoint actually consumes. Upserts the result under
       ``/workbench/endpoints/<endpoint_name>/input_columns``.

    2. **Explicit (``input_cols`` provided):** stores the caller-supplied
       list as-is. Use when the model's declared features() list is wrong
       or incomplete.

    Idempotent — re-running refreshes the stored list if the model's
    feature contract changed.

    Args:
        endpoint: A Workbench ``Endpoint`` instance deployed from a
            Model/FeatureSet.
        input_cols: Optional explicit list of input column names. When
            provided, skips auto-discovery.

    Returns:
        list[str]: The input columns that were registered.

    Raises:
        ValueError: If ``input_cols`` is provided but empty or contains
            non-string entries.
        RuntimeError: In auto-discovery mode, if the endpoint has no input
            model, or the model has no declared features.
    """
    if input_cols is not None:
        if not input_cols:
            raise ValueError("register_input_columns: input_cols must be a non-empty list")
        if not all(isinstance(c, str) for c in input_cols):
            raise ValueError("register_input_columns: input_cols must contain only strings")
        sorted_cols = sorted(input_cols)
        key = input_columns_key(endpoint.name)
        ParameterStore().upsert(key, sorted_cols)
        log.info(
            f"register_input_columns: registered {len(sorted_cols)} columns for "
            f"{endpoint.name} at {key} (explicit)"
        )
        return sorted_cols

    from workbench.core.artifacts import ModelCore

    model = ModelCore(endpoint.get_input())
    if not model.exists():
        raise RuntimeError(
            f"register_input_columns: endpoint {endpoint.name} has no input model — "
            f"register_input_columns only applies to endpoints deployed from a Model/FeatureSet."
        )
    features = model.features()
    if not features:
        raise RuntimeError(
            f"register_input_columns: model '{model.name}' has no declared features."
        )

    sorted_cols = sorted(features)
    key = input_columns_key(endpoint.name)
    ParameterStore().upsert(key, sorted_cols)
    log.info(f"register_input_columns: registered {len(sorted_cols)} columns for {endpoint.name} at {key}")
    return sorted_cols


def get_output_columns(endpoint_name: str) -> Optional[List[str]]:
    """Look up the output columns registered for an endpoint.

    Args:
        endpoint_name: e.g. ``"smiles-to-2d-v1"`` or ``"abalone-regression"``.

    Returns:
        List of output column names, or ``None`` if the endpoint hasn't
        registered its output columns yet. Call :func:`register_output_columns`
        from the endpoint's deploy script to populate the list.
    """
    return ParameterStore().get(output_columns_key(endpoint_name))


def register_output_columns(endpoint, output_cols: Optional[List[str]] = None) -> List[str]:
    """Register an endpoint's output columns in ParameterStore.

    Works for any endpoint that emits new columns during inference:
    feature endpoints emit computed feature columns (descriptors,
    fingerprints, etc.); predictor endpoints emit prediction / confidence /
    quantile columns.

    Two modes:

    1. **Auto-discovery (default, ``output_cols=None``):** runs a small smoke
       inference with only the columns the model declares as inputs (plus the
       FeatureSet's id column), diffs the output to find the added columns,
       filters out diagnostic and provenance columns, and upserts the result
       under ``/workbench/endpoints/<endpoint_name>/output_columns``.

    2. **Explicit (``output_cols`` provided):** skips the smoke inference
       entirely and registers the caller-supplied list as-is. Use when you
       already have the canonical column list (e.g. from the model script
       that produced the endpoint) and want deterministic registration
       without the inference round-trip.

    Downstream model-training scripts can then look up the columns this
    endpoint produces without re-running inference to diff:

        >>> ParameterStore().get("/workbench/endpoints/smiles-to-2d-v1/output_columns")

    Idempotent — re-running refreshes the stored list if the endpoint's
    output shape changed.

    Auto-discovery relies on the following conventions:
        1. The input FeatureSet has an ``id_column`` (standard Workbench).
        2. The Model's ``features()`` lists the columns the endpoint actually
           consumes (e.g. ``["smiles"]`` for a feature endpoint, or the
           training feature set for a predictor endpoint). Only those are
           passed to the endpoint during the smoke test — this stops output
           columns from being masked when the source FeatureSet happens to
           contain columns with the same names.
        3. Columns whose name starts with ``desc`` are treated as
           diagnostic/telemetry (e.g. ``desc3d_status``,
           ``desc3d_compute_time_s``) and excluded from the registered list.
        4. Columns in :data:`NON_FEATURE_COLUMNS` (preprocessing provenance:
           ``orig_smiles``, ``salt``, ``undefined_chiral_centers``) are also
           excluded.

    Explicit mode trusts the caller — the list is stored verbatim (sorted for
    consistency) with no filtering. It's the caller's job to pass the right set.

    Args:
        endpoint: A Workbench ``Endpoint`` instance deployed from a
            Model/FeatureSet (auto-routes async transparently).
        output_cols: Optional explicit list of output column names to
            register. When provided, skips auto-discovery.

    Returns:
        list[str]: The output columns that were registered.

    Raises:
        ValueError: If ``output_cols`` is provided but empty or contains
            non-string entries.
        RuntimeError: In auto-discovery mode, if the endpoint has no input
            model / input FeatureSet, the model has no declared features,
            or the inference produces no new columns at all (i.e. the
            endpoint just echoes its input).
    """
    if output_cols is not None:
        if not output_cols:
            raise ValueError("register_output_columns: output_cols must be a non-empty list")
        if not all(isinstance(c, str) for c in output_cols):
            raise ValueError("register_output_columns: output_cols must contain only strings")
        sorted_cols = sorted(output_cols)
        key = output_columns_key(endpoint.name)
        ParameterStore().upsert(key, sorted_cols)
        log.info(
            f"register_output_columns: registered {len(sorted_cols)} columns for "
            f"{endpoint.name} at {key} (explicit — no smoke inference)"
        )
        return sorted_cols

    from workbench.core.artifacts import FeatureSetCore, ModelCore

    model = ModelCore(endpoint.get_input())
    if not model.exists():
        raise RuntimeError(
            f"register_output_columns: endpoint {endpoint.name} has no input model — "
            f"register_output_columns only applies to endpoints deployed from a Model/FeatureSet."
        )
    fs = FeatureSetCore(model.get_input())
    if not fs.exists():
        raise RuntimeError(
            f"register_output_columns: input FeatureSet '{model.get_input()}' not found for endpoint {endpoint.name}."
        )
    model_inputs = model.features()
    if not model_inputs:
        raise RuntimeError(
            f"register_output_columns: model '{model.name}' has no declared features — "
            f"cannot build a minimal smoke-test input."
        )

    # Build a *minimal* smoke-test input: only the columns the model actually
    # consumes, plus the FeatureSet's id column. If we passed the full FeatureSet,
    # any output column whose name already exists in the input would be silently
    # masked by the diff. 5 rows is enough to diff — pass limit=5 so we don't
    # pull the whole FeatureSet to Athena just to slice off the top.
    keep = [fs.id_column] + [c for c in model_inputs if c != fs.id_column]
    sample_df = fs.pull_dataframe(limit=5)[keep].copy()

    input_cols = set(sample_df.columns)
    result_df = endpoint.inference(sample_df)
    added = [c for c in result_df.columns if c not in input_cols]

    # Drop non-output columns. Two buckets:
    #   - Diagnostics (prefix `desc*`): per-row telemetry like `desc3d_status`,
    #     `desc3d_compute_time_s`, etc.
    #   - Preprocessing provenance (:data:`NON_FEATURE_COLUMNS`): columns that
    #     record the preprocessing pipeline's view of the input (e.g.
    #     `orig_smiles`, `salt`, `undefined_chiral_centers`).
    def is_output(c: str) -> bool:
        return not c.startswith("desc") and c not in NON_FEATURE_COLUMNS

    output_cols = sorted(c for c in added if is_output(c))
    dropped = sorted(c for c in added if not is_output(c))
    if dropped:
        log.info(
            f"register_output_columns: filtered {len(dropped)} non-output column(s) "
            f"(desc* / NON_FEATURE_COLUMNS): {dropped}"
        )

    if not output_cols:
        raise RuntimeError(
            f"register_output_columns: {endpoint.name} produced no output columns after "
            f"diagnostic filtering — the endpoint emitted only the input columns it "
            f"received plus optional desc*/provenance columns."
        )

    key = output_columns_key(endpoint.name)
    ParameterStore().upsert(key, output_cols)
    log.info(f"register_output_columns: registered {len(output_cols)} columns for {endpoint.name} at {key}")
    return output_cols


if __name__ == "__main__":
    """Exercise the Endpoint Utilities"""

    # Create an Endpoint
    endpoint_name = "abalone-regression"
    my_endpoint = Endpoint(endpoint_name)
    if not my_endpoint.exists():
        print(f"Endpoint {endpoint_name} does not exist.")
        exit(1)

    # Get the Model Data URL
    model_data_url = internal_model_data_url(my_endpoint.endpoint_config_name(), my_endpoint.boto3_session)
    print(model_data_url)

    # Get the training data
    my_train_df = get_training_data(my_endpoint)
    print(my_train_df)

    # Get the evaluation data
    my_eval_df = get_evaluation_data(my_endpoint)
    print(my_eval_df)

    # Backtrack to the FeatureSet
    my_fs = backtrack_to_fs(my_endpoint)
    print(my_fs)

    # Also test for realtime endpoints
    rt_endpoint = Endpoint("abalone-regression-end-rt")
    if rt_endpoint.exists():
        model_data_url = internal_model_data_url(rt_endpoint.endpoint_config_name(), rt_endpoint.boto3_session)
        print(model_data_url)

    # Check if the endpoint is monitored
    is_monitored_status = is_monitored(my_endpoint.name, my_endpoint.sm_client)
    print(f"Is the endpoint '{my_endpoint.name}' monitored? {is_monitored_status}")
