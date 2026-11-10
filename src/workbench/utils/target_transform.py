"""Target transforms: train in transformed space, serve predictions on the original scale.

A model's ``target_transform`` hyperparameter names an invertible, monotonic map. Training
applies ``forward`` to the target column(s); inference applies ``inverse`` to predictions.
Metrics are always computed on the original scale, so a transform never changes how a model
is reported or compared.

Monotonicity is the contract: rank-based metrics (Spearman, Kendall) are identical with and
without a transform, and only placement-dependent metrics move.

A transform provides ``name``, ``forward``, ``inverse`` and ``__repr__``; is monotonic
increasing; takes JSON-serializable constructor arguments, since
``resolve_target_transform`` rebuilds it from ``hyperparameters.json`` inside the serving
container; and raises on out-of-domain input at training time rather than returning NaN.
Transforms live here rather than behind a plugin hook: plugins are not on the endpoint
import path, so a plugin-supplied transform would train and then fail to load at deployment.

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

    ``offset`` shifts the target away from zero, and the assay's LLOQ is where it belongs:
    below that threshold the assay cannot separate values, so resolution spent there is spent
    on noise. What matters is the offset relative to the target's own range, not its absolute
    size — over a target confined to [0, 1], an offset of 1.0 gives the bottom decile 14% of
    the output range against a linear map's 10%, which is barely a transform at all.
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


class Logit:
    """``log10(y / (ceiling - y))``, for a target bounded at both ends.

    A percent-type target — oral bioavailability, percent unbound — is bounded at 0 and at
    100, and ``log10`` resolves only the low end: over %F it leaves the top decile 2.6% of the
    output range, so the model stops separating 84% from 100%. A logit keeps both tails, and
    ``ceiling`` sets the balance. Decile shares on %F, bottom/top, against a linear map's
    7.7%/16.4%: 54.6%/8.3% at ceiling 120, 48.3%/15.3% at 105, 39.9%/28.6% at 100.5.

    A ceiling above the physical bound leaves every measured value defined; one at the bound
    spends more range on the top tail but rejects anything reaching it. ``floor`` lifts
    sub-LLOQ values, as ``Log10``'s ``offset`` does, and a negative target is out of domain
    whatever the floor.

    ``inverse`` lands in ``(0, ceiling)`` by construction, so a logit model needs no
    ``prediction_bounds``. Bounds still apply after the inverse, so a high bound below the
    ceiling clips real predictions into a mass point.
    """

    name = "logit"

    def __init__(self, ceiling: float, floor: float = 0.0):
        self.ceiling = float(ceiling)
        self.floor = float(floor)
        if not np.isfinite(self.ceiling) or not np.isfinite(self.floor):
            raise ValueError(
                f"target_transform 'logit' needs a finite ceiling and floor, got ceiling {ceiling!r} "
                f"and floor {floor!r}"
            )
        if not 0.0 <= self.floor < self.ceiling:
            raise ValueError(
                f"target_transform 'logit' needs 0 <= floor < ceiling, got floor {self.floor:g} "
                f"and ceiling {self.ceiling:g}"
            )

    def forward(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float)
        known = np.isfinite(values)

        too_low = known & (values < 0.0) if self.floor > 0.0 else known & (values <= 0.0)
        if too_low.any():
            requirement = "target >= 0" if self.floor > 0.0 else "target > 0"
            remedy = (
                "A negative target has no bound fraction a logit can represent, so drop those rows."
                if self.floor > 0.0
                else "Set a floor just under the assay's lower limit."
            )
            raise ValueError(
                f"target_transform 'logit' needs {requirement}, but {int(too_low.sum())} of "
                f"{int(known.sum())} values are not (smallest target: {float(np.min(values[known])):g}, "
                f"floor: {self.floor:g}). {remedy}"
            )

        at_ceiling = known & (values >= self.ceiling)
        if at_ceiling.any():
            raise ValueError(
                f"target_transform 'logit' needs target < ceiling, but {int(at_ceiling.sum())} of "
                f"{int(known.sum())} values are not (largest target: {float(np.max(values[known])):g}, "
                f"ceiling: {self.ceiling:g}). Cap or drop those rows — raising the ceiling to cover "
                f"them spends the target's range on values the assay cannot support."
            )

        clamped = np.maximum(values, self.floor) if self.floor > 0.0 else values
        odds = np.full(clamped.shape, np.nan, dtype=float)
        np.divide(clamped, self.ceiling - clamped, out=odds, where=known)
        result = np.full(clamped.shape, np.nan, dtype=float)
        np.log10(odds, out=result, where=known)
        return result

    def inverse(self, predictions: np.ndarray) -> np.ndarray:
        # Sigmoid form with a clipped exponent, so an extreme prediction saturates inside
        # (0, ceiling) rather than overflowing to a NaN
        exponent = np.clip(-np.asarray(predictions, dtype=float), -300.0, 300.0)
        return self.ceiling / (1.0 + np.power(10.0, exponent))

    def __repr__(self) -> str:
        return f"{self.name}(ceiling={self.ceiling:g}, floor={self.floor:g})"


class ArcSinh:
    """``asinh(y / scale)``, logarithmic above ``scale`` and linear below it.

    Defined for every real target, so it has no domain error and no offset to place against
    the assay's floor: exact zeros, sub-LLOQ noise and negative readings all pass through,
    which is what makes it usable where ``Log10`` is not — logD and anything else that goes
    below zero.

    ``scale`` is where resolution stops being spent, the role ``Log10``'s ``offset`` plays,
    and the two land in the same place: on a 0-1 target the bottom decile gets 29.4% of the
    output range under ``scale`` 0.1 against 28.9% under ``offset`` 0.1. What matters is
    ``scale`` against the target's own range — share of the output range given to the bottom
    decile, where a linear map gives 10%:

    | target range | scale 0.01 | scale 0.1 | scale 1 | scale 10 |
    |---|---|---|---|---|
    | 0 - 1 | 56.6% | 29.4% | 11.3% | 10.0% |
    | 0 - 100 | 76.7% | 69.7% | 56.6% | 29.4% |

    So the default of 1.0 is a plain ``asinh(y)``: a strong transform on a percentage, barely
    a transform on a fraction.
    """

    name = "arcsinh"

    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"target_transform 'arcsinh' needs a finite scale above zero, got {scale!r}")

    def forward(self, y: np.ndarray) -> np.ndarray:
        return np.arcsinh(np.asarray(y, dtype=float) / self.scale)

    def inverse(self, predictions: np.ndarray) -> np.ndarray:
        return self.scale * np.sinh(np.asarray(predictions, dtype=float))

    def __repr__(self) -> str:
        return f"{self.name}(scale={self.scale:g})"


TRANSFORMS = {cls.name: cls for cls in (Identity, Log10, Logit, ArcSinh)}
Transform = Union[Identity, Log10, Logit, ArcSinh]


def resolve_target_transform(spec: TransformSpec) -> Transform:
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


def clip_quantile_columns(df, bounds: Optional[Sequence[Optional[float]]], prefix: str = "q_") -> None:
    """Clip UQ prediction quantiles into the same range as the point prediction, in place.

    A quantile predicts the target and the target obeys the bound, so clipping one cannot
    change interval coverage — it only drops mass the model has declared impossible. The
    spread in ``prediction_std`` is a dispersion estimate that UQ calibrates against, so it
    stays unclipped and ``prediction ± prediction_std`` can still reach past a bound.

    Args:
        df: Frame of UQ columns, modified in place.
        bounds: ``(low, high)``, either side optional, or None to leave the frame untouched.
        prefix: Names the quantile columns.
    """
    if bounds is None:
        return
    for column in df.columns:
        if column.startswith(prefix):
            df[column] = clip_to_bounds(df[column].to_numpy(), bounds)


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
