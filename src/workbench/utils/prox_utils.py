"""Helpers for building and retrieving compound Proximity models.

Keeps the construction and precomputed-lookup logic out of the Model and FeatureSet
API classes — they just handle caching and pass through the right arguments.
"""

import copy
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd

from workbench.utils.target_transform import Identity, resolve_target_transforms

if TYPE_CHECKING:
    from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
    from workbench.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity


def build_proximity(
    df: pd.DataFrame,
    space: str,
    id_column: str,
    feature_list: Optional[list] = None,
    target: Optional[str] = None,
    include_all_columns: bool = False,
) -> "Union[FingerprintProximity, FeatureSpaceProximity]":
    """Construct a proximity model over ``df`` for the given space.

    Args:
        df: Reference DataFrame.
        space: ``"fingerprint"`` (Tanimoto over SMILES/fingerprints) or ``"features"``
            (standardized Euclidean over numeric features).
        id_column: Identifier column.
        feature_list: Numeric feature columns — required for ``"features"``, ignored for
            ``"fingerprint"``.
        target: Optional target column surfaced in neighbor results.
        include_all_columns: Include all DataFrame columns in neighbor results.

    Returns:
        A FingerprintProximity or FeatureSpaceProximity.
    """
    if space not in ("fingerprint", "features"):
        raise ValueError(f"space must be 'fingerprint' or 'features', got {space!r}")
    if space == "features" and not feature_list:
        raise ValueError("space='features' requires feature_list=[...]")

    if space == "fingerprint":
        from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

        return FingerprintProximity(df, id_column=id_column, target=target, include_all_columns=include_all_columns)

    from workbench.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity

    return FeatureSpaceProximity(
        df, id_column=id_column, features=feature_list, target=target, include_all_columns=include_all_columns
    )


def rescale_proximity_targets(prox, hyperparameters: dict, targets, to_model_space: bool):
    """A proximity whose target columns are moved between the original scale and the model's space.

    UQ works in the model's ``target_transform`` space, so the proximity embedded in its
    artifact carries model-space targets, while a proximity built from a FeatureSet carries
    original-scale ones. This converts one into the other.

    Args:
        prox: The proximity to convert.
        hyperparameters: The model's hyperparameters, which name its ``target_transform``.
        targets: The model's target column(s). A proximity may carry only some of them.
        to_model_space: True forwards original-scale targets; False inverts model-space ones.

    Returns:
        A shallow copy with converted target columns, or ``prox`` itself for an
        untransformed model.
    """
    targets = targets if isinstance(targets, list) else [targets]
    transforms = resolve_target_transforms((hyperparameters or {}).get("target_transform"), targets)
    active = {t: x for t, x in transforms.items() if not isinstance(x, Identity) and t in prox.df.columns}
    if not active:
        return prox

    converted = copy.copy(prox)
    converted.df = prox.df.copy()
    for target, transform in active.items():
        values = converted.df[target].to_numpy(dtype=float)
        converted.df[target] = transform.forward(values, strict=False) if to_model_space else transform.inverse(values)
    return converted


def precomputed_model_proximity(model, space: str):
    """Return the proximity embedded in a model's artifact for ``space``, or None.

    Only fingerprint proximity is embedded today (carried in the UQ artifact);
    feature-space proximity is never embedded, so ``"features"`` always returns None.
    The embedded targets are in the model's space, so they are inverted on the way out.
    """
    if space == "features":
        return None
    try:
        uq = model.uq_model()
    except FileNotFoundError:
        return None
    prox = getattr(uq, "prox", None)
    if prox is None:
        return None
    return rescale_proximity_targets(prox, model.hyperparameters(), model.target(), to_model_space=False)


def fingerprint_prox_model_local(
    model,
    include_all_columns: bool = False,
    radius: int = 2,
    n_bits: int = 4096,
) -> "FingerprintProximity":
    """Create a FingerprintProximity over a Model's full FeatureSet, marking training rows.

    Note: FingerprintProximity auto-detects binary vs. count fingerprints from the
    fingerprint column format (comma-separated → count, otherwise binary).

    Args:
        model (Model): The Model used to create the fingerprint proximity model.
        include_all_columns (bool): Include all DataFrame columns in neighbor results.
        radius (int): Morgan fingerprint radius (default: 2).
        n_bits (int): Number of bits for the fingerprint (default: 4096).

    Returns:
        FingerprintProximity: proximity over the full FeatureSet with an ``in_model`` flag.
    """
    from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
    from workbench.api import FeatureSet

    target = model.target()

    fs = FeatureSet(model.get_input())
    id_column = fs.id_column

    # Build over the full FeatureSet, marking which rows were in the model's training set
    full_df = fs.pull_dataframe()
    model_ids = set(model.training_view().pull_dataframe()[id_column])
    full_df["in_model"] = full_df[id_column].isin(model_ids)

    return FingerprintProximity(
        full_df,
        id_column=id_column,
        target=target,
        include_all_columns=include_all_columns,
        radius=radius,
        n_bits=n_bits,
    )
