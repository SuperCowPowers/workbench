"""Shared regression-UQ training/inference helpers used by all model templates.

The chemprop / pytorch / xgb model templates all do the same dance for
regression UQ:

    1. Build a fingerprint-proximity reference set from the training data.
    2. Fit ``UQModelV0`` and ``UQModelV1`` on the same validation predictions
       and ensemble std.
    3. Save both artifacts into the model bundle.
    4. At inference (``model_fn``), load whichever version
       ``hyperparameters["uq_version"]`` selects (default ``"v0"``).

That logic lives here so each template can call:

    # ---- Training ----
    uq_dict = fit_regression_uq(...)
    uq_out = uq_dict["uq_model"].predict(...)         # active for df_val cols
    save_regression_uq(uq_dict, args.model_dir)       # writes both V0 and V1

    # ---- Inference (model_fn) ----
    uq_model = load_regression_uq(model_dir)          # returns just the active
    return {..., "uq_model": uq_model, ...}

For offline comparison of the non-active version, callers use
``Model.uq_model(version="v0"|"v1")`` — that loads either version explicitly
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

log = logging.getLogger("workbench")


def _normalize_version(version: Optional[str]) -> str:
    """Coerce a UQ version string to canonical 'v0'/'v1' form, defaulting to 'v0'."""
    if version is None:
        return "v0"
    v = str(version).strip().lower()
    if v not in ("v0", "v1"):
        raise ValueError(f"Unknown UQ version '{version}' (expected 'v0' or 'v1')")
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
    """Fit UQModelV0 + UQModelV1 on the same validation slice.

    Args:
        y_true: True target values for the validation set, shape (n,).
        y_pred: Predicted values (ensemble mean), shape (n,).
        y_std: Ensemble standard deviation, shape (n,).
        val_ids: List of compound IDs aligned with the above arrays.
        prox_df: DataFrame for the V1 FingerprintProximity reference set.
            Must contain ``id_column``, a ``smiles`` column, and the target
            column. Typically built from the full training set (CV rows
            marked ``in_model=True``).
        id_column: Name of the ID column in ``prox_df``.
        target: Name of the target column in ``prox_df``.
        active_version: Which version is the "primary" one (``"v0"`` or
            ``"v1"``). Doesn't affect what gets fit (both always do); only
            determines which is returned under the ``uq_model`` key.

    Returns:
        dict with keys ``uq_model`` (the active instance), ``v0``, ``v1``.
    """
    active = _normalize_version(active_version)

    log.info("Fitting UQModelV0 (isotonic on prediction+std) ...")
    uq_model_v0 = UQModelV0.fit(y_true, y_pred, y_std)

    log.info("Fitting UQModelV1 (proximity-augmented RF error model) ...")
    prox = FingerprintProximity(prox_df, id_column=id_column, target=target)
    uq_model_v1 = UQModelV1(prox)
    uq_model_v1.fit(val_ids, y_true, y_pred, y_std)

    uq_model_active = uq_model_v1 if active == "v1" else uq_model_v0
    log.info(f"Active UQ version for training-time df_val columns: {active}")

    return {"uq_model": uq_model_active, "v0": uq_model_v0, "v1": uq_model_v1}


def save_regression_uq(uq_dict: Optional[dict], model_dir: str) -> None:
    """Save V0 and V1 artifacts from a fit_regression_uq() result.

    ``None`` is a no-op so classification-task code paths can pass it
    unconditionally without first building an empty placeholder dict.
    """
    if uq_dict is None:
        return
    if uq_dict.get("v0") is not None:
        uq_dict["v0"].save(model_dir)
    if uq_dict.get("v1") is not None:
        uq_dict["v1"].save(model_dir)


def load_regression_uq(model_dir: str) -> Optional[Union[UQModelV0, UQModelV1]]:
    """Load the active regression UQ model from a bundle.

    Reads ``hyperparameters.json["uq_version"]`` to decide which version is
    active (defaults to ``"v0"``), then loads that one. Falls back to the
    other if the requested version's artifact isn't present. Returns ``None``
    if neither is in the bundle (e.g. a classification model).

    For explicit offline access to a specific version (or to both for
    comparison), use ``Model.uq_model(version=...)`` instead.
    """
    v0_present = os.path.exists(os.path.join(model_dir, UQModelV0.METADATA_FILENAME))
    v1_present = os.path.exists(os.path.join(model_dir, "uq_model.joblib"))
    if not (v0_present or v1_present):
        return None

    bundle_hp_path = os.path.join(model_dir, "hyperparameters.json")
    bundle_hp = {}
    if os.path.exists(bundle_hp_path):
        with open(bundle_hp_path) as fp:
            bundle_hp = json.load(fp)
    active_version = _normalize_version(bundle_hp.get("uq_version", "v0"))

    if active_version == "v1" and v1_present:
        return UQModelV1.load(model_dir)
    if v0_present:
        return UQModelV0.load(model_dir)
    # active was v0 but no v0 artifact → fall back to v1
    return UQModelV1.load(model_dir)
