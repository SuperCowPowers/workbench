"""Admin + lookup helpers for *feature endpoints* — endpoints that take an
input column (typically ``smiles``) and emit computed feature columns
(descriptors, fingerprints, conformer-derived metrics, etc.).

Feature endpoints register their output columns in ParameterStore under the
convention:

    /workbench/feature_lists/<endpoint_name>

Write side (admin, from a deploy script):

    from workbench.utils.feature_endpoint_utils import register_features
    register_features(end)   # idempotent — re-registers if the output shape changed

Read side (downstream model-training scripts):

    from workbench.utils.feature_endpoint_utils import get_endpoint_features
    feature_cols = get_endpoint_features("smiles-to-2d-v1")
"""

from __future__ import annotations

import logging
from typing import List, Optional

from workbench.api import ParameterStore

log = logging.getLogger("workbench")

FEATURE_LIST_PREFIX = "/workbench/feature_lists"

# Provenance / preprocessing-metadata columns that feature endpoints emit
# alongside real features. Excluded from the registered feature list by
# `register_features`. Add a column here only if it's truly not a feature
# (e.g. a record of the original SMILES before canonicalization).
NON_FEATURE_COLUMNS = frozenset(
    {
        "orig_smiles",
        "salt",
        "undefined_chiral_centers",
    }
)


def feature_list_key(endpoint_name: str) -> str:
    """Return the ParameterStore key under which an endpoint's feature
    columns are registered.

    Example:
        >>> feature_list_key("smiles-to-2d-v1")
        '/workbench/feature_lists/smiles-to-2d-v1'
    """
    return f"{FEATURE_LIST_PREFIX}/{endpoint_name}"


def ensure_demo_featureset(name: str = "feature_endpoint_fs"):
    """Ensure the shared demo FeatureSet (used as the smoke-test / training
    source for all feature endpoints) exists. Backed by the public AqSol
    dataset.

    Feature endpoints consume SMILES and produce computed descriptors — they
    don't "learn" from training data. The demo FeatureSet just gives workbench
    something to hang the Model / Endpoint artifacts off of during creation.

    Idempotent — returns the existing FeatureSet if it's already there.
    """
    # Lazy imports to keep this utility module cheap and avoid any circular
    # import risk if someone imports feature_endpoint_utils from deep in core.
    from workbench.api import FeatureSet, PublicData
    from workbench.core.transforms.pandas_transforms import PandasToFeatures

    fs = FeatureSet(name)
    if fs.exists():
        return fs

    aqsol = PublicData().get("comp_chem/aqsol/aqsol_public_data")
    aqsol.columns = aqsol.columns.str.lower()
    to_features = PandasToFeatures(name)
    to_features.set_input(aqsol, id_column="id")
    to_features.set_output_tags(["aqsol", "public"])
    to_features.transform()
    fs = FeatureSet(name)
    fs.set_owner("FeatureEndpoint")
    return fs


def get_endpoint_features(endpoint_name: str) -> Optional[List[str]]:
    """Look up the feature columns registered for a feature endpoint.

    Args:
        endpoint_name: e.g. ``"smiles-to-2d-v1"``.

    Returns:
        List of feature column names, or ``None`` if the endpoint hasn't
        registered its features yet. Call :func:`register_features` from
        the endpoint's deploy script to populate the list.
    """
    return ParameterStore().get(feature_list_key(endpoint_name))


def register_features(endpoint, feature_cols: Optional[List[str]] = None) -> List[str]:
    """Register a feature endpoint's output columns in ParameterStore.

    Two modes:

    1. **Auto-discovery (default, ``feature_cols=None``):** runs a small smoke
       inference with only the columns the model declares as inputs (plus the
       FeatureSet's id column), diffs the output to find the added columns,
       filters out diagnostic and provenance columns, and upserts the result
       under ``/workbench/feature_lists/<endpoint_name>``.

    2. **Explicit (``feature_cols`` provided):** skips the smoke inference
       entirely and registers the caller-supplied list as-is. Use when you
       already have the canonical feature list (e.g. from the model script
       that produced the endpoint) and want deterministic registration
       without the inference round-trip.

    Downstream model-training scripts can then look up the feature set this
    endpoint produces without re-running inference to diff columns:

        >>> ParameterStore().get("/workbench/feature_lists/smiles-to-2d-v1")

    Idempotent — re-running refreshes the stored list if the endpoint's
    output shape changed.

    Auto-discovery relies on the following conventions:
        1. The input FeatureSet has an ``id_column`` (standard Workbench).
        2. The Model's ``features()`` lists the columns the endpoint actually
           consumes (e.g. ``["smiles"]``). Only those are passed to the
           endpoint during the smoke test — this stops output columns from
           being masked when the source FeatureSet happens to contain
           pre-baked descriptor columns with the same names.
        3. Columns whose name starts with ``desc`` are treated as
           diagnostic/telemetry (e.g. ``desc3d_status``,
           ``desc3d_compute_time_s``) and excluded from the registered
           feature list.
        4. Columns in :data:`NON_FEATURE_COLUMNS` (preprocessing provenance:
           ``orig_smiles``, ``salt``, ``undefined_chiral_centers``) are also
           excluded.

    Explicit mode trusts the caller — the list is stored verbatim (sorted for
    consistency) with no filtering. It's the caller's job to pass the right set.

    Args:
        endpoint: A Workbench ``Endpoint`` instance deployed from a
            Model/FeatureSet (auto-routes async transparently).
        feature_cols: Optional explicit list of feature column names to
            register. When provided, skips auto-discovery.

    Returns:
        list[str]: The feature columns that were registered.

    Raises:
        ValueError: If ``feature_cols`` is provided but empty or contains
            non-string entries.
        RuntimeError: In auto-discovery mode, if the endpoint has no input
            model / input FeatureSet, the model has no declared feature_list,
            or the inference produces no new non-diagnostic columns.
    """
    # Explicit-list mode: trust the caller, just persist.
    if feature_cols is not None:
        if not feature_cols:
            raise ValueError("register_features: feature_cols must be a non-empty list")
        if not all(isinstance(c, str) for c in feature_cols):
            raise ValueError("register_features: feature_cols must contain only strings")
        sorted_cols = sorted(feature_cols)
        key = feature_list_key(endpoint.name)
        ParameterStore().upsert(key, sorted_cols)
        log.info(
            f"register_features: registered {len(sorted_cols)} columns for "
            f"{endpoint.name} at {key} (explicit — no smoke inference)"
        )
        return sorted_cols

    # Auto-discovery mode — run smoke inference and diff.
    # Local imports to avoid pulling core classes into this module's top-level
    # imports (and to keep this utility file cheap to import).
    from workbench.core.artifacts import FeatureSetCore, ModelCore

    model = ModelCore(endpoint.get_input())
    if not model.exists():
        raise RuntimeError(
            f"register_features: endpoint {endpoint.name} has no input model — "
            f"register_features only applies to endpoints deployed from a Model/FeatureSet."
        )
    fs = FeatureSetCore(model.get_input())
    if not fs.exists():
        raise RuntimeError(
            f"register_features: input FeatureSet '{model.get_input()}' not found for endpoint {endpoint.name}."
        )
    model_inputs = model.features()
    if not model_inputs:
        raise RuntimeError(
            f"register_features: model '{model.name}' has no declared feature_list — "
            f"cannot build a minimal smoke-test input."
        )

    # Build a *minimal* smoke-test input: only the columns the model actually
    # consumes, plus the FeatureSet's id column. This is the key fix for the
    # "AqSol pre-baked descriptors" problem — if we passed the full FeatureSet,
    # any output column whose name already exists in the input would be
    # silently masked by the diff. 5 rows is enough to diff — pass limit=5 so
    # we don't pull the whole FeatureSet to Athena just to slice off the top.
    keep = [fs.id_column] + [c for c in model_inputs if c != fs.id_column]
    sample_df = fs.pull_dataframe(limit=5)[keep].copy()

    # Run inference and diff columns to find what the endpoint adds.
    input_cols = set(sample_df.columns)
    result_df = endpoint.inference(sample_df)
    added = [c for c in result_df.columns if c not in input_cols]

    # Drop non-feature columns. Two buckets:
    #   - Diagnostics (prefix `desc*`): per-row telemetry like `desc3d_status`,
    #     `desc3d_compute_time_s`, etc.
    #   - Preprocessing provenance (:data:`NON_FEATURE_COLUMNS`): columns that
    #     record the preprocessing pipeline's view of the input (e.g.
    #     `orig_smiles`, `salt`, `undefined_chiral_centers`).
    def is_feature(c: str) -> bool:
        return not c.startswith("desc") and c not in NON_FEATURE_COLUMNS

    feature_cols = sorted(c for c in added if is_feature(c))
    dropped = sorted(c for c in added if not is_feature(c))
    if dropped:
        log.info(
            f"register_features: filtered {len(dropped)} non-feature column(s) "
            f"(desc* / NON_FEATURE_COLUMNS): {dropped}"
        )

    if not feature_cols:
        raise RuntimeError(
            f"register_features: {endpoint.name} produced no feature columns after "
            f"diagnostic filtering. This function is only for feature endpoints "
            f"(SMILES → descriptors etc.); predictor endpoints should not call it."
        )

    key = feature_list_key(endpoint.name)
    ParameterStore().upsert(key, feature_cols)
    log.info(f"register_features: registered {len(feature_cols)} columns for {endpoint.name} at {key}")
    return feature_cols
