"""Shared regression-UQ training/inference helpers used by all model templates.

The chemprop / pytorch / xgb model templates all do the same dance for
regression UQ:

    1. Build a fingerprint-proximity reference set from the training data.
    2. Fit ``UQModelV0``, ``UQModelV1``, and ``UQModelV2`` on the same
       validation predictions and ensemble std (V2 needs only the proximity).
    3. Save all three artifacts into the model bundle (V1 and V2 share
       ``uq_proximity.joblib``).
    4. At inference (``model_fn``), load whichever version
       ``hyperparameters["uq_version"]`` selects (default ``"v0"``).

That logic lives here so each template can call:

    # ---- Training ----
    uq_dict = fit_regression_uq(...)
    uq_out = uq_dict["uq_model"].predict(...)         # active for df_val cols
    save_regression_uq(uq_dict, args.model_dir)       # writes V0, V1, V2

    # ---- Inference (model_fn) ----
    uq_model = load_regression_uq(model_dir)          # returns just the active
    return {..., "uq_model": uq_model, ...}

For offline comparison of non-active versions, callers use
``Model.uq_model(version="v0"|"v1"|"v2")`` — that loads any version explicitly
without going through the endpoint.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Union

from workbench.endpoints.fingerprint_proximity import FingerprintProximity
from workbench.endpoints.uq_model_v0 import UQModelV0
from workbench.endpoints.uq_model_v1 import UQModelV1
from workbench.endpoints.uq_model_v2 import UQModelV2

log = logging.getLogger("workbench")

_VALID_VERSIONS = ("v0", "v1", "v2")


def _normalize_version(version: Optional[str]) -> str:
    """Coerce a UQ version string to canonical form, defaulting to 'v0'."""
    if version is None:
        return "v0"
    v = str(version).strip().lower()
    if v not in _VALID_VERSIONS:
        raise ValueError(f"Unknown UQ version '{version}' (expected one of {_VALID_VERSIONS})")
    return v


def fit_regression_uq(
    *,
    y_true,
    y_pred,
    y_std,
    val_ids: list,
    prox_df,
    id_column: str,
    target: str,
    active_version: str = "v0",
) -> dict:
    """Fit UQModelV0, UQModelV1, and UQModelV2 on the same training slice.

    V0 and V1 use the validation predictions/std for calibration. V2 uses only
    the proximity reference set and doesn't need ensemble predictions, but it's
    fit here alongside V0/V1 so the bundle always carries all three.

    Args:
        y_true: True target values for the validation set, shape (n,).
        y_pred: Predicted values (ensemble mean), shape (n,).
        y_std: Ensemble standard deviation, shape (n,).
        val_ids: List of compound IDs aligned with the above arrays.
        prox_df: DataFrame for the V1/V2 FingerprintProximity reference set.
            Must contain ``id_column``, a ``smiles`` column, and the target
            column. Typically built from the full training set (CV rows
            marked ``in_model=True``).
        id_column: Name of the ID column in ``prox_df``.
        target: Name of the target column in ``prox_df``.
        active_version: Which version is the "primary" one (``"v0"``, ``"v1"``,
            or ``"v2"``). Doesn't affect what gets fit (all three always do);
            only determines which is returned under the ``uq_model`` key.

    Returns:
        dict with keys ``uq_model`` (the active instance), ``v0``, ``v1``, ``v2``.
    """
    active = _normalize_version(active_version)

    log.info("Fitting UQModelV0 (isotonic on prediction+std) ...")
    uq_model_v0 = UQModelV0.fit(y_true, y_pred, y_std)

    # V1 and V2 share the same FingerprintProximity instance — built once.
    prox = FingerprintProximity(prox_df, id_column=id_column, target=target)

    log.info("Fitting UQModelV1 (proximity-augmented RF error model) ...")
    uq_model_v1 = UQModelV1(prox)
    uq_model_v1.fit(val_ids, y_true, y_pred, y_std)

    log.info("Fitting UQModelV2 (applicability-domain from proximity) ...")
    uq_model_v2 = UQModelV2.fit(prox)

    active_lookup = {"v0": uq_model_v0, "v1": uq_model_v1, "v2": uq_model_v2}
    uq_model_active = active_lookup[active]
    log.info(f"Active UQ version for training-time df_val columns: {active}")

    return {
        "uq_model": uq_model_active,
        "v0": uq_model_v0,
        "v1": uq_model_v1,
        "v2": uq_model_v2,
    }


def save_regression_uq(uq_dict: Optional[dict], model_dir: str) -> None:
    """Save V0, V1, V2 artifacts from a fit_regression_uq() result.

    ``None`` is a no-op so classification-task code paths can pass it
    unconditionally without first building an empty placeholder dict.

    V1 and V2 share ``uq_proximity.joblib``. V1 is saved first so its
    proximity file is on disk when V2.save() checks for it; V2 then skips
    rewriting the shared file.
    """
    if uq_dict is None:
        return
    if uq_dict.get("v0") is not None:
        uq_dict["v0"].save(model_dir)
    if uq_dict.get("v1") is not None:
        uq_dict["v1"].save(model_dir)
    if uq_dict.get("v2") is not None:
        uq_dict["v2"].save(model_dir)


def load_regression_uq(model_dir: str) -> Optional[Union[UQModelV0, UQModelV1, UQModelV2]]:
    """Load the active regression UQ model from a bundle.

    Reads ``hyperparameters.json["uq_version"]`` to decide which version is
    active (defaults to ``"v0"``), then loads that one. Falls back to any
    other available version if the requested one's artifact isn't present.
    Returns ``None`` if no UQ artifacts are in the bundle (e.g. a
    classification model).

    For explicit offline access to a specific version, use
    ``Model.uq_model(version=...)`` instead.
    """
    available = {
        "v0": os.path.exists(os.path.join(model_dir, UQModelV0.METADATA_FILENAME)),
        "v1": os.path.exists(os.path.join(model_dir, "uq_model.joblib")),
        "v2": os.path.exists(os.path.join(model_dir, UQModelV2.METADATA_FILENAME)),
    }
    if not any(available.values()):
        return None

    bundle_hp_path = os.path.join(model_dir, "hyperparameters.json")
    bundle_hp = {}
    if os.path.exists(bundle_hp_path):
        with open(bundle_hp_path) as fp:
            bundle_hp = json.load(fp)
    active_version = _normalize_version(bundle_hp.get("uq_version", "v0"))

    # Try active version first; if missing, fall back in v0 → v1 → v2 order.
    order = [active_version] + [v for v in _VALID_VERSIONS if v != active_version]
    loaders = {"v0": UQModelV0.load, "v1": UQModelV1.load, "v2": UQModelV2.load}
    for v in order:
        if available[v]:
            return loaders[v](model_dir)
    return None
