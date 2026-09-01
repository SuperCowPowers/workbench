"""A per-target capture must carry that target's own predictions.

Metrics for a multi-target capture are computed from `{target}_pred`, so the predictions
saved beside them have to point at the same column. Otherwise a capture cannot reproduce
its own metrics, and any per-row analysis (paired bootstrap, residual plots) silently
reads the primary target's values.
"""

import pandas as pd
import pytest

from workbench.core.artifacts.endpoint_core import EndpointCore

remap = EndpointCore._remap_multi_target_columns


@pytest.fixture
def multi_target_df():
    return pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "cyp3a4_pred": [5.0, 6.0, 7.0],
            "cyp3a4_pred_std": [0.1, 0.2, 0.3],
            "cyp3a4_confidence": [0.9, 0.8, 0.7],
            "cyp2c9_pred": [1.0, 2.0, 3.0],
            "cyp2c9_pred_std": [0.4, 0.5, 0.6],
            "cyp2c9_confidence": [0.6, 0.5, 0.4],
            # what the endpoint emits: the standard columns alias the primary target
            "prediction": [5.0, 6.0, 7.0],
            "prediction_std": [0.1, 0.2, 0.3],
            "confidence": [0.9, 0.8, 0.7],
        }
    )


def test_remaps_a_non_primary_target(multi_target_df):
    out = remap(multi_target_df, "cyp2c9")
    assert out["prediction"].tolist() == [1.0, 2.0, 3.0]
    assert out["prediction_std"].tolist() == [0.4, 0.5, 0.6]
    assert out["confidence"].tolist() == [0.6, 0.5, 0.4]


def test_primary_target_is_unchanged(multi_target_df):
    out = remap(multi_target_df, "cyp3a4")
    assert out["prediction"].tolist() == [5.0, 6.0, 7.0]


def test_does_not_mutate_the_input(multi_target_df):
    remap(multi_target_df, "cyp2c9")
    assert multi_target_df["prediction"].tolist() == [5.0, 6.0, 7.0]


def test_single_target_frame_passes_through():
    """A single-target model has no `{target}_pred` columns; leave `prediction` alone."""
    df = pd.DataFrame({"id": ["a"], "prediction": [4.2], "prediction_std": [0.1]})
    out = remap(df, "solubility")
    assert out["prediction"].tolist() == [4.2]
    assert out["prediction_std"].tolist() == [0.1]


def test_missing_per_target_column_is_dropped_not_leaked():
    """A target with no confidence of its own must not inherit the primary's.

    Leaving it would label the primary target's confidence with another target's
    capture name — wrong rather than merely missing, and impossible to spot downstream.
    """
    df = pd.DataFrame({"id": ["a"], "cyp2c9_pred": [1.0], "prediction": [5.0], "confidence": [0.9]})
    out = remap(df, "cyp2c9")
    assert out["prediction"].tolist() == [1.0]
    assert "confidence" not in out.columns


def test_capture_can_reproduce_its_own_metric(multi_target_df):
    """The defect this guards: scoring the capture must match scoring {target}_pred."""
    truth = pd.Series([1.2, 2.2, 3.2])
    out = remap(multi_target_df, "cyp2c9")
    from_capture = (out["prediction"] - truth).abs().mean()
    from_target = (multi_target_df["cyp2c9_pred"] - truth).abs().mean()
    assert from_capture == pytest.approx(from_target)


def test_per_target_uq_columns_are_remapped():
    """Confidence and the interval columns follow their target, same as prediction."""
    df = pd.DataFrame(
        {
            "id": ["a"],
            "cyp3a4_pred": [5.0],
            "cyp3a4_confidence": [0.9],
            "cyp3a4_q_05": [4.0],
            "cyp2c9_pred": [1.0],
            "cyp2c9_confidence": [0.4],
            "cyp2c9_q_05": [0.5],
            "prediction": [5.0],
            "confidence": [0.9],
            "q_05": [4.0],
        }
    )
    out = remap(df, "cyp2c9")
    assert out["prediction"].tolist() == [1.0]
    assert out["confidence"].tolist() == [0.4]
    assert out["q_05"].tolist() == [0.5]
