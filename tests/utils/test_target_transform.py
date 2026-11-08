"""Tests for target transforms."""

import json

import numpy as np
import pytest

from workbench.utils.target_transform import (
    Identity,
    Log10,
    clip_to_bounds,
    load_target_transform,
    resolve_prediction_bounds,
    resolve_target_transform,
    resolve_target_transforms,
)

TARGETS = ["ksol", "logd", "mppb"]


def test_resolve_accepts_none_bare_name_and_dict():
    assert isinstance(resolve_target_transform(None), Identity)
    assert isinstance(resolve_target_transform("identity"), Identity)
    assert isinstance(resolve_target_transform("log10"), Log10)
    resolved = resolve_target_transform({"name": "log10", "offset": 0.25})
    assert isinstance(resolved, Log10)
    assert resolved.offset == 0.25


def test_resolve_defaults_offset_to_zero():
    """A bare log10 is a plain log10, not log10(y + 1)."""
    assert resolve_target_transform("log10").offset == 0.0


def test_resolve_raises_on_typo():
    """A hyperparameter typo must fail at training time, not train on a raw target."""
    with pytest.raises(ValueError, match="Unknown target_transform"):
        resolve_target_transform("log-10")


def test_resolve_raises_on_bad_arguments():
    with pytest.raises(ValueError, match="Bad arguments"):
        resolve_target_transform({"name": "log10", "offsset": 1.0})


def test_resolve_raises_on_malformed_spec():
    with pytest.raises(ValueError, match="must be None, a name, or a dict"):
        resolve_target_transform({"offset": 1.0})


def test_identity_round_trip():
    y = np.array([-2.0, 0.0, 3.5])
    transform = Identity()
    np.testing.assert_allclose(transform.inverse(transform.forward(y)), y)


def test_log10_round_trip():
    y = np.array([0.001, 0.1, 1.0, 250.0])
    transform = Log10()
    np.testing.assert_allclose(transform.inverse(transform.forward(y)), y, rtol=1e-12)


def test_log10_round_trip_with_offset():
    y = np.array([0.0, 0.5, 10.0])
    transform = Log10(offset=0.01)
    np.testing.assert_allclose(transform.inverse(transform.forward(y)), y, atol=1e-12)


def test_log10_forward_values():
    transform = Log10()
    np.testing.assert_allclose(transform.forward(np.array([1.0, 10.0, 100.0])), [0.0, 1.0, 2.0])


def test_log10_offset_of_one_is_near_linear_on_a_fraction():
    """Why offset defaults to 0: log10(1 + p) barely stretches a 0-1 target."""
    p = np.array([0.001, 0.01, 0.1])
    stretched = Log10(offset=0.001).forward(p)
    flat = Log10(offset=1.0).forward(p)
    assert np.ptp(stretched) > 10 * np.ptp(flat)


def test_log10_raises_outside_its_domain():
    with pytest.raises(ValueError, match="needs target \\+ offset > 0"):
        Log10().forward(np.array([0.0, 1.0, 10.0]))


def test_log10_offset_brings_zeros_into_the_domain():
    result = Log10(offset=0.5).forward(np.array([0.0, 1.0]))
    assert np.isfinite(result).all()


def test_log10_passes_nan_through():
    """Missing and censored targets stay missing rather than raising."""
    result = Log10().forward(np.array([1.0, np.nan, 100.0]))
    assert np.isnan(result[1])
    np.testing.assert_allclose(result[[0, 2]], [0.0, 2.0])


def test_transforms_preserve_order():
    """The monotonicity contract: rank metrics are unaffected by a transform."""
    y = np.array([0.002, 0.04, 0.3, 7.0, 900.0])
    forward = Log10(offset=0.001).forward(y)
    assert np.all(np.diff(forward) > 0)


def test_spec_round_trips_through_the_registry():
    transform = Log10(offset=0.01)
    assert resolve_target_transform(transform.spec()).offset == transform.offset


def test_multi_target_single_spec_applies_to_every_target():
    resolved = resolve_target_transforms({"name": "log10", "offset": 0.01}, TARGETS)
    assert set(resolved) == set(TARGETS)
    assert all(isinstance(t, Log10) and t.offset == 0.01 for t in resolved.values())


def test_multi_target_bare_name_and_none():
    assert all(isinstance(t, Log10) for t in resolve_target_transforms("log10", TARGETS).values())
    assert all(isinstance(t, Identity) for t in resolve_target_transforms(None, TARGETS).values())


def test_multi_target_mapping_is_per_target():
    resolved = resolve_target_transforms({"ksol": "log10", "mppb": {"name": "log10", "offset": 0.5}}, TARGETS)
    assert isinstance(resolved["ksol"], Log10) and resolved["ksol"].offset == 0.0
    assert isinstance(resolved["mppb"], Log10) and resolved["mppb"].offset == 0.5
    assert isinstance(resolved["logd"], Identity), "omitted targets get identity"


def test_multi_target_mapping_rejects_unknown_columns():
    """A typo'd target column must fail rather than silently transform nothing."""
    with pytest.raises(ValueError, match="not targets"):
        resolve_target_transforms({"ksoll": "log10"}, TARGETS)


def test_multi_target_name_key_wins_over_mapping_reading():
    """A target column literally named 'name' is not a case we support; 'name' means a spec."""
    resolved = resolve_target_transforms({"name": "identity"}, TARGETS)
    assert all(isinstance(t, Identity) for t in resolved.values())


def test_resolve_prediction_bounds_shared_and_per_target():
    assert resolve_prediction_bounds(None, TARGETS) == {t: None for t in TARGETS}
    assert resolve_prediction_bounds((0.0, 100.0), TARGETS) == {t: (0.0, 100.0) for t in TARGETS}
    per_target = resolve_prediction_bounds({"mppb": (0.0, 1.0)}, TARGETS)
    assert per_target == {"ksol": None, "logd": None, "mppb": (0.0, 1.0)}


def test_resolve_prediction_bounds_validates():
    with pytest.raises(ValueError, match="not targets"):
        resolve_prediction_bounds({"nope": (0.0, 1.0)}, TARGETS)
    with pytest.raises(ValueError, match="must be \\(low, high\\)"):
        resolve_prediction_bounds((0.0, 1.0, 2.0), TARGETS)
    with pytest.raises(ValueError, match="low must be below high"):
        resolve_prediction_bounds({"ksol": (1.0, 0.0)}, TARGETS)


def _write_hyperparameters(model_dir, hyperparameters):
    (model_dir / "hyperparameters.json").write_text(json.dumps(hyperparameters))
    return str(model_dir)


def test_load_target_transform_single_target(tmp_path):
    model_dir = _write_hyperparameters(
        tmp_path, {"target_transform": {"name": "log10", "offset": 0.01}, "prediction_bounds": [0.0, 1.0]}
    )
    transform, bounds = load_target_transform(model_dir)
    assert isinstance(transform, Log10) and transform.offset == 0.01
    assert bounds == (0.0, 1.0)


def test_load_target_transform_multi_target(tmp_path):
    model_dir = _write_hyperparameters(tmp_path, {"target_transform": {"ksol": "log10"}})
    transforms, bounds = load_target_transform(model_dir, TARGETS)
    assert isinstance(transforms["ksol"], Log10)
    assert isinstance(transforms["logd"], Identity)
    assert bounds == {t: None for t in TARGETS}


def test_load_target_transform_without_hyperparameters_is_identity(tmp_path):
    """A model trained before transforms existed carries no spec and must still serve."""
    transform, bounds = load_target_transform(str(tmp_path))
    assert isinstance(transform, Identity)
    assert bounds is None


def test_clip_to_bounds():
    values = np.array([-0.5, 0.25, 1.5])
    np.testing.assert_allclose(clip_to_bounds(values, (0.0, 1.0)), [0.0, 0.25, 1.0])
    np.testing.assert_allclose(clip_to_bounds(values, (0.0, None)), [0.0, 0.25, 1.5])
    np.testing.assert_allclose(clip_to_bounds(values, (None, 1.0)), [-0.5, 0.25, 1.0])
    np.testing.assert_allclose(clip_to_bounds(values, None), values)


def test_clip_to_bounds_validates():
    with pytest.raises(ValueError, match="must be \\(low, high\\)"):
        clip_to_bounds(np.array([1.0]), (0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="low must be below high"):
        clip_to_bounds(np.array([1.0]), (1.0, 0.0))
