"""Shared regression-UQ training/inference helpers used by all model templates.

The chemprop / pytorch / xgb model templates all do the same dance for
regression UQ:

    1. Build a proximity reference set from the training data — fingerprint-based
       when a 'smiles' column is present, otherwise over the model's features.
    2. Fit ``UQModelV0``, ``UQModelV1``, and ``UQModelV2`` on the same
       out-of-fold predictions and ensemble std (V2 needs only the proximity).
    3. Save all three artifacts into the model bundle (V1 and V2 share
       ``uq_proximity.joblib``).
    4. At inference (``model_fn``), load whichever version the bundle's
       ``hyperparameters["uq_version"]`` selects.

That logic lives here so each template can call:

    # ---- Training ----
    uq_dict = fit_regression_uq(...)
    uq_out = uq_dict["uq_model"].predict(...)         # active for df_oof cols
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

from workbench.endpoints.feature_space_proximity import FeatureSpaceProximity
from workbench.endpoints.fingerprint_proximity import FingerprintProximity
from workbench.endpoints.proximity import Proximity
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


def _build_proximity(prox_df, *, id_column: str, target: str, features: Optional[list] = None) -> Optional[Proximity]:
    """Pick the neighbor backend for V1/V2 from what the reference set carries.

    A 'smiles' column wins — structure-based neighborhoods are the stronger signal.
    Otherwise fall back to the model's own feature columns. Returns None when
    neither is available, which leaves V1/V2 unfit.
    """
    if prox_df is None:
        return None

    if "smiles" in prox_df.columns:
        log.info("Building FingerprintProximity reference set ('smiles') ...")
        return FingerprintProximity(prox_df, id_column=id_column, target=target)

    usable = [f for f in (features or []) if f in prox_df.columns]
    if not usable:
        return None

    log.info(f"No 'smiles' column: building FeatureSpaceProximity over {len(usable)} feature columns ...")
    return FeatureSpaceProximity(prox_df, id_column=id_column, features=usable, target=target)


def uq_query_df(uq_model, df, features: Optional[list] = None):
    """Build the neighbor-lookup query frame a UQ model's proximity backend expects.

    V0 needs no query payload, so it returns None. V1/V2 ask their own backend what
    it indexes on — 'smiles'/'fingerprint' for FingerprintProximity, the feature
    columns for FeatureSpaceProximity — which keeps callers out of the business of
    sniffing columns. Returns None when ``df`` can't satisfy the backend, and the
    caller should skip UQ scoring for that batch.

    Args:
        uq_model: A fit UQ model (V0, V1, or V2), or None.
        df: The inference batch.
        features: Model feature columns, used when the batch carries feature values
            under different names than the proximity's own (rare; defaults to the
            proximity's feature list).

    Returns:
        A DataFrame slice to hand to ``uq_model.predict()``, or None.
    """
    prox = getattr(uq_model, "prox", None)
    if prox is None:
        return None  # V0 (or no model): scores from prediction + std alone

    if prox.space == "fingerprint":
        cols = [c for c in ("smiles", "fingerprint") if c in df.columns]
        return df[cols] if cols else None

    # Feature space: the scaler needs every column it was fit on, so this is all-or-nothing
    needed = features or prox.features
    missing = [f for f in needed if f not in df.columns]
    if missing:
        log.warning(f"UQ skipped: batch is missing {len(missing)} proximity feature(s), e.g. {missing[:3]}")
        return None
    return df[needed]


def fit_regression_uq(
    *,
    y_true,
    y_pred,
    y_std,
    oof_ids: list,
    prox_df=None,
    id_column: str,
    target: str,
    features: Optional[list] = None,
    active_version: str = "v1",
) -> dict:
    """Fit the regression UQ models on the out-of-fold training predictions.

    V0 (isotonic calibration on prediction+std) is always fit. V1 and V2 are
    neighborhood models built on a ``Proximity`` backend, chosen by what ``prox_df``
    carries: a ``smiles`` column gives structure-based ``FingerprintProximity``,
    otherwise ``features`` gives ``FeatureSpaceProximity`` over the model's own
    feature columns. V1/V2 are skipped only when neither is available.

    Future me: the prox_df construction is currently duplicated in the
    xgb/pytorch/chemprop templates. Consider passing the training df here and
    building the reference set in this one place so the templates just call this.

    Args:
        y_true: True target values for the out-of-fold rows, shape (n,).
        y_pred: Out-of-fold predicted values (ensemble mean), shape (n,).
        y_std: Ensemble standard deviation, shape (n,).
        oof_ids: Compound IDs aligned with the above arrays.
        prox_df: DataFrame for the V1/V2 proximity reference set, or None. Must
            contain ``id_column``, the target column, and either a ``smiles``
            column or the ``features`` columns (CV rows marked ``in_model=True``).
            When None, only V0 is fit.
        id_column: Name of the ID column in ``prox_df``.
        target: Name of the target column in ``prox_df``.
        features: Model feature columns, used to build a FeatureSpaceProximity when
            ``prox_df`` has no ``smiles`` column.
        active_version: Which version is the "primary" one (``"v0"``, ``"v1"``,
            or ``"v2"``), defaulting to ``"v1"``. If the requested version
            wasn't fit, falls back to v0 with a warning.

    Returns:
        dict with keys ``uq_model`` (the active instance), ``v0``, ``v1``, ``v2``
        (``v1``/``v2`` are None when no proximity backend could be built).
    """
    active = _normalize_version(active_version)

    log.info("Fitting UQModelV0 (isotonic on prediction+std) ...")
    uq_model_v0 = UQModelV0.fit(y_true, y_pred, y_std)

    uq_model_v1 = None
    uq_model_v2 = None
    prox = _build_proximity(prox_df, id_column=id_column, target=target, features=features)
    if prox is not None:
        log.info("Fitting UQModelV1 (proximity-augmented RF error model) ...")
        uq_model_v1 = UQModelV1(prox)
        uq_model_v1.fit(oof_ids, y_true, y_pred, y_std)

        log.info("Fitting UQModelV2 (applicability-domain from proximity) ...")
        uq_model_v2 = UQModelV2.fit(prox)

    active_lookup = {"v0": uq_model_v0, "v1": uq_model_v1, "v2": uq_model_v2}
    uq_model_active = active_lookup.get(active)
    if uq_model_active is None:
        log.warning(f"UQ '{active}' needs a proximity reference set ('smiles' or features); falling back to v0")
        uq_model_active = uq_model_v0
    log.info(f"Active UQ version for training-time df_oof columns: {active}")

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
