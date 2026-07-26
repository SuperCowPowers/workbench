"""Fast unit tests for ``workbench.training.pytorch_core``.

Frame alignment plus a small end-to-end run of the shared fold recipe — the same
``train_pytorch_fold`` the template publishes with and the HPO objective scores with.
"""

import pandas as pd
import pytest

from workbench.training.pytorch_core import PyTorchFoldSpec, align_frame, train_pytorch_fold


def _spec(**overrides):
    base = dict(
        hyperparameters={},
        target="y",
        continuous_cols=["x1"],
        categorical_cols=["color"],
        category_mappings={"color": ["red", "blue"]},
        orig_features=["x1", "color", "fingerprint"],
        compressed_features=["fingerprint"],
        cat_impute_values={"color": "red"},
    )
    base.update(overrides)
    return PyTorchFoldSpec(**base)


def _raw_frame():
    return pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "color": ["blue", None, "green"],  # a NaN and an unseen category
            "fingerprint": ["101", "011", "110"],
            "y": [0.1, 0.2, 0.3],
        }
    )


def test_align_frame_applies_fitted_transforms_to_a_raw_frame():
    """A raw holdout gets the mappings, decompression, and NaN imputation training used."""
    df = _raw_frame()
    out = align_frame(_spec(), df)

    assert str(out["color"].dtype) == "category"
    # NaN and the unseen category both resolve to the imputation value.
    assert out["color"].tolist() == ["blue", "red", "red"]
    # Bitstring decompressed to per-bit columns; the compressed column is gone.
    assert "fingerprint" not in out.columns
    assert {"fin_0", "fin_1", "fin_2"} <= set(out.columns)
    # The input frame is not mutated.
    assert "fingerprint" in df.columns and df["color"].dtype == object


def test_align_frame_is_idempotent():
    """The training frame has already been through the template's preprocessing, and the
    runner aligns both frames — a second pass must be a no-op."""
    once = align_frame(_spec(), _raw_frame())
    twice = align_frame(_spec(), once)
    pd.testing.assert_frame_equal(once, twice)


def test_align_frame_without_alignment_state_passes_through():
    """A spec with no fitted state (unit-test scale) aligns to a plain copy."""
    spec = _spec(category_mappings={}, orig_features=None, compressed_features=None, cat_impute_values=None)
    df = _raw_frame()
    out = align_frame(spec, df)
    pd.testing.assert_frame_equal(out, df)
    assert out is not df


def _train_spec():
    torch = pytest.importorskip("torch")  # noqa: F841
    hp = {
        "layers": "8-4",
        "dropout": 0.0,
        "max_epochs": 2,
        "early_stopping_patience": 2,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "loss": "L1Loss",
        "restore_best_weights": False,
        "seed": 42,
    }
    return PyTorchFoldSpec(
        hyperparameters=hp,
        target="y",
        continuous_cols=["x1", "x2"],
        categorical_cols=[],
        verbose=False,
    )


def test_train_pytorch_fold_is_seed_deterministic():
    """Two runs at the same fold_idx produce identical weights — the parity the shared
    recipe guarantees between a scored trial and the published model."""
    import torch

    x, y = torch.randn(24, 2), torch.randn(24, 1)
    spec = _train_spec()
    model_a, _ = train_pytorch_fold(spec, (x, None, y), (x, None, y), fold_idx=0)
    model_b, _ = train_pytorch_fold(spec, (x, None, y), (x, None, y), fold_idx=0)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)


def test_fold_idx_offsets_the_seed_for_ensemble_diversity():
    import torch

    x, y = torch.randn(24, 2), torch.randn(24, 1)
    spec = _train_spec()
    model_a, _ = train_pytorch_fold(spec, (x, None, y), (x, None, y), fold_idx=0)
    model_b, _ = train_pytorch_fold(spec, (x, None, y), (x, None, y), fold_idx=1)
    assert any(not torch.equal(pa, pb) for pa, pb in zip(model_a.parameters(), model_b.parameters()))
