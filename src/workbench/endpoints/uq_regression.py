"""Shared regression-UQ training/inference helpers used by all model templates.

The chemprop / pytorch / xgb model templates all do the same dance for
regression UQ:

    1. Build a fingerprint-proximity reference set from the training data.
    2. Fit ``UQModelV0`` and ``UQModelV1`` on the same validation predictions
       and ensemble std.
    3. Save both artifacts into the model bundle.
    4. At inference (``model_fn``), load whichever artifacts exist and pick an
       active version from ``hyperparameters["uq_version"]`` (default ``"v0"``).

That logic lives here so each template can call:

    # ---- Training ----
    uq_dict = fit_regression_uq(
        y_true=y_val_true,
        y_pred=y_val_pred,
        y_std=y_val_std,
        val_ids=val_ids,
        prox_df=prox_df,
        id_column=id_column,
        target=target,
        active_version=hyperparameters.get("uq_version", "v0"),
    )
    # apply active UQ to df_val columns ...
    save_regression_uq(uq_dict, args.model_dir)

    # ---- Inference (model_fn) ----
    uq_dict = load_regression_uq(model_dir)
    # model_dict["uq_model"] is the active one; uq_model_v0 / uq_model_v1 are
    # both retained for comparison.

Returned dicts always carry the same keys regardless of which version was
active: ``uq_model`` (active), ``uq_model_v0``, ``uq_model_v1``,
``active_uq_version``. Templates plug the whole dict into the model_fn return
value so predict_fn can keep using ``model_dict["uq_model"]`` unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

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
            ``"v1"``). Returned in the result dict; doesn't affect what gets
            fit (both always do).

    Returns:
        dict with keys: ``uq_model`` (the active instance), ``uq_model_v0``,
        ``uq_model_v1``, ``active_uq_version``.
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

    return {
        "uq_model": uq_model_active,
        "uq_model_v0": uq_model_v0,
        "uq_model_v1": uq_model_v1,
        "active_uq_version": active,
    }


def save_regression_uq(uq_dict: dict, model_dir: str) -> None:
    """Save both V0 and V1 artifacts from a fit_regression_uq() result."""
    if uq_dict.get("uq_model_v0") is not None:
        uq_dict["uq_model_v0"].save(model_dir)
    if uq_dict.get("uq_model_v1") is not None:
        uq_dict["uq_model_v1"].save(model_dir)


def load_regression_uq(model_dir: str) -> dict:
    """Load V0 + V1 UQ artifacts from a model bundle and pick the active one.

    The active version is read from ``hyperparameters.json["uq_version"]`` in
    ``model_dir`` (defaults to ``"v0"`` if missing). If the active version's
    artifact isn't present, falls back to the other one. If neither is present,
    every returned value is ``None``.

    Args:
        model_dir: Directory containing the model artifacts.

    Returns:
        dict with keys: ``uq_model`` (active or None), ``uq_model_v0``,
        ``uq_model_v1``, ``active_uq_version``.
    """
    uq_model_v0 = None
    uq_model_v1 = None
    if os.path.exists(os.path.join(model_dir, UQModelV0.METADATA_FILENAME)):
        uq_model_v0 = UQModelV0.load(model_dir)
    if os.path.exists(os.path.join(model_dir, "uq_model.joblib")):
        uq_model_v1 = UQModelV1.load(model_dir)

    bundle_hp_path = os.path.join(model_dir, "hyperparameters.json")
    bundle_hp = {}
    if os.path.exists(bundle_hp_path):
        with open(bundle_hp_path) as fp:
            bundle_hp = json.load(fp)
    active_version = _normalize_version(bundle_hp.get("uq_version", "v0"))

    if active_version == "v1" and uq_model_v1 is not None:
        active = uq_model_v1
    elif uq_model_v0 is not None:
        active = uq_model_v0
    else:
        # Either V1-only bundle, or no UQ at all.
        active = uq_model_v1

    return {
        "uq_model": active,
        "uq_model_v0": uq_model_v0,
        "uq_model_v1": uq_model_v1,
        "active_uq_version": active_version,
    }
