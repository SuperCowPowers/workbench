"""Target transforms: train in transformed space, serve predictions on the original scale.

A model's ``target_transform`` hyperparameter names an invertible, monotonic map. Training
applies ``forward`` to the target column(s); inference applies ``inverse`` to predictions.
Metrics are always computed on the original scale, so a transform never changes how a model
is reported or compared.

Monotonicity is the contract: rank-based metrics (Spearman, Kendall) are identical with and
without a transform, and only placement-dependent metrics move.

numpy-only by design — this module is on the endpoint import path.
"""

import json
import os
from typing import Optional, Sequence, Union

import numpy as np

TransformSpec = Union[None, str, dict]


class Identity:
    """No-op transform. The default for every model."""

    name = "identity"

    def forward(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float)

    def inverse(self, predictions: np.ndarray) -> np.ndarray:
        return np.asarray(predictions, dtype=float)

    def __repr__(self) -> str:
        return self.name


class Log10:
    """``log10(y + offset)``, for targets spanning orders of magnitude.

    ``offset`` shifts the target away from zero and belongs near the assay's lower limit. An
    offset large relative to the target's own range flattens the map toward linear — over a
    target confined to [0, 1], an offset of 1.0 is barely a transform at all.
    """

    name = "log10"

    def __init__(self, offset: float = 0.0):
        self.offset = float(offset)
        if not np.isfinite(self.offset):
            raise ValueError(f"target_transform 'log10' needs a finite offset, got {offset!r}")

    def forward(self, y: np.ndarray) -> np.ndarray:
        shifted = np.asarray(y, dtype=float) + self.offset
        known = np.isfinite(shifted)
        out_of_domain = known & (shifted <= 0.0)
        if out_of_domain.any():
            smallest = float(np.min(shifted[known])) - self.offset
            raise ValueError(
                f"target_transform 'log10' needs target + offset > 0, but {int(out_of_domain.sum())} "
                f"of {int(known.sum())} values are not (smallest target: {smallest:g}, "
                f"offset: {self.offset:g}). Raise the offset to just under the assay's lower limit."
            )
        result = np.full(shifted.shape, np.nan, dtype=float)
        np.log10(shifted, out=result, where=known)
        return result

    def inverse(self, predictions: np.ndarray) -> np.ndarray:
        return np.power(10.0, np.asarray(predictions, dtype=float)) - self.offset

    def __repr__(self) -> str:
        return f"{self.name}(offset={self.offset:g})"


TRANSFORMS = {cls.name: cls for cls in (Identity, Log10)}


def resolve_target_transform(spec: TransformSpec) -> Union[Identity, Log10]:
    """Build a transform from a hyperparameter value.

    Accepts ``None`` (identity), a bare name (``"log10"``), or a name with arguments
    (``{"name": "log10", "offset": 0.001}``). Raises on anything else so a typo fails at
    training time rather than silently training on an untransformed target.

    Args:
        spec: The ``target_transform`` hyperparameter value.

    Returns:
        The resolved transform.
    """
    if spec is None:
        return Identity()
    if isinstance(spec, tuple(TRANSFORMS.values())):
        return spec
    if isinstance(spec, str):
        spec = {"name": spec}
    if not isinstance(spec, dict) or "name" not in spec:
        raise ValueError(
            f"target_transform must be None, a name, or a dict with a 'name' key, got {spec!r}. "
            f"Available transforms: {sorted(TRANSFORMS)}"
        )

    params = dict(spec)
    name = params.pop("name")
    if name not in TRANSFORMS:
        raise ValueError(f"Unknown target_transform {name!r}. Available transforms: {sorted(TRANSFORMS)}")
    try:
        return TRANSFORMS[name](**params)
    except TypeError as exc:
        raise ValueError(f"Bad arguments for target_transform {name!r}: {exc}") from exc


def _is_per_target_mapping(spec: TransformSpec, targets: Sequence[str]) -> bool:
    """True when every key names a target column, so the dict is a per-target mapping.

    Tested before the single-spec form, so a model with a target column called ``name`` reads
    ``{"name": ...}`` as that target rather than as a transform name.
    """
    return isinstance(spec, dict) and bool(spec) and all(key in targets for key in spec)


def resolve_target_transforms(spec: TransformSpec, targets: Sequence[str]) -> dict:
    """Resolve a transform for each target of a multi-target model.

    A dict whose every key is a target column is a ``{target_column: spec}`` mapping, and
    targets it omits get identity — multi-target models carry targets in different units, so
    a shared transform rarely suits all of them. Anything else is a single spec (``None``,
    ``"log10"``, or ``{"name": "log10", ...}``) applied to every target.

    Args:
        spec: The ``target_transform`` hyperparameter value.
        targets: The model's target columns.

    Returns:
        ``{target_column: transform}`` covering every target.
    """
    if _is_per_target_mapping(spec, targets):
        return {target: resolve_target_transform(spec.get(target)) for target in targets}
    if isinstance(spec, dict) and spec and "name" not in spec:
        unknown = [column for column in spec if column not in targets]
        raise ValueError(f"target_transform names columns that are not targets: {unknown}. Targets: {list(targets)}")

    shared = resolve_target_transform(spec)
    return {target: shared for target in targets}


def resolve_prediction_bounds(spec, targets: Sequence[str]) -> dict:
    """Resolve prediction bounds for each target of a multi-target model.

    A single ``(low, high)`` applies to every target; a dict is a ``{target_column: (low,
    high)}`` mapping, and targets it omits are unbounded.

    Args:
        spec: The ``prediction_bounds`` hyperparameter value.
        targets: The model's target columns.

    Returns:
        ``{target_column: bounds_or_None}`` covering every target.
    """
    if spec is None:
        return {target: None for target in targets}
    if isinstance(spec, dict):
        unknown = [column for column in spec if column not in targets]
        if unknown:
            raise ValueError(
                f"prediction_bounds names columns that are not targets: {unknown}. Targets: {list(targets)}"
            )
        return {target: _validated_bounds(spec.get(target)) for target in targets}
    return {target: _validated_bounds(spec) for target in targets}


def load_target_transform(model_dir: str, targets: Optional[Sequence[str]] = None) -> tuple:
    """Resolve the transform and bounds a trained model was fitted with.

    Reads the model's own ``hyperparameters.json``, so inference inverts exactly what
    training applied rather than trusting a separately-passed value.

    Args:
        model_dir: The model directory, as handed to ``model_fn``.
        targets: Target columns for a multi-target model. Omit for a single target.

    Returns:
        ``(transform, bounds)`` for a single target, or ``({target: transform}, {target:
        bounds})`` when ``targets`` is given.
    """
    hyperparameters = {}
    hyperparameters_path = os.path.join(model_dir, "hyperparameters.json")
    if os.path.exists(hyperparameters_path):
        with open(hyperparameters_path, encoding="utf-8") as file_handle:
            hyperparameters = json.load(file_handle)

    spec = hyperparameters.get("target_transform")
    bounds = hyperparameters.get("prediction_bounds")
    if targets is None:
        return resolve_target_transform(spec), _validated_bounds(bounds)
    return resolve_target_transforms(spec, targets), resolve_prediction_bounds(bounds, targets)


def _validated_bounds(bounds: Optional[Sequence[Optional[float]]]) -> Optional[tuple]:
    """Check a ``(low, high)`` pair, so a malformed bound fails at training time."""
    if bounds is None:
        return None
    if isinstance(bounds, str) or len(bounds) != 2:
        raise ValueError(f"prediction_bounds must be (low, high), got {bounds!r}")
    low, high = bounds
    if low is not None and high is not None and low >= high:
        raise ValueError(f"prediction_bounds low must be below high, got {bounds!r}")
    return (low, high)


def clip_to_bounds(values: np.ndarray, bounds: Optional[Sequence[Optional[float]]]) -> np.ndarray:
    """Clip predictions into a physically valid range.

    Independent of any transform — a non-negative or percentage target wants this under the
    identity transform too. Either side may be ``None`` to leave it unbounded. Values pile up
    as a mass point at a bound, destroying ordering among them, so bound only targets whose
    predictions should not reach the bound.

    Args:
        values: Predictions on the original scale.
        bounds: ``(low, high)``, either side optional, or ``None`` to leave values untouched.

    Returns:
        The clipped predictions.
    """
    if bounds is None:
        return np.asarray(values, dtype=float)
    low, high = _validated_bounds(bounds)
    return np.clip(np.asarray(values, dtype=float), low, high)
