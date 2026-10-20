"""Tests for the Atom-JEPA fold trainer.

These fine-tune a real (tiny) model on CPU, so they need the ``modeling`` extra and the
Atom-JEPA checkpoint (local cache or the foundation registry).
"""

import numpy as np
import pytest

pytest.importorskip("atom_jepa")

# Workbench Imports
from workbench.training.atomjepa_core import FoldModel, build_conformers, lr_scale, train_fold  # noqa: E402

pytestmark = pytest.mark.medium

SMILES = [
    "CC(=O)Nc1ccc(O)cc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "O=C(O)c1ccccc1OC(=O)C",
    "c1ccc2ccccc2c1",
    "Cc1ccccc1N",
    "COc1ccc(CCN)cc1",
    "O=C1CCCN1",
    "c1ccncc1",
    "CCOC(=O)c1ccccc1",
    "Clc1ccc(Cl)cc1",
    "OC1CCCCC1",
    "c1ccc(-c2ccccc2)cc1",
]
HYPERPARAMETERS = {
    "epochs": 2,
    "batch_size": 4,
    "encoder_lr": 5e-5,
    "head_lr": 5e-4,
    "head_weight_decay": 0.1,
    "min_lr": 1e-7,
    "warmup_epochs": 1,
    "freeze_encoder_epochs": 1,
    "ema_decay": 0.99,
    "grad_clip": 10.0,
    "dropout": 0.1,
    "selection_conformers": 1,
}


@pytest.fixture(scope="module")
def fitted():
    conformers = build_conformers(SMILES, 2)
    rng = np.random.default_rng(0)
    y = rng.normal(3.0, 1.0, len(SMILES))
    extra = rng.normal(size=(len(SMILES), 2))
    fold, val_pred = train_fold(
        HYPERPARAMETERS, conformers[:8], y[:8], conformers[8:], y[8:], extra[:8], extra[8:], seed=0
    )
    return fold, val_pred, conformers, extra


def test_lr_scale_warms_up_then_decays():
    hp = {"warmup_epochs": 4, "epochs": 20, "freeze_encoder_epochs": 0, "min_lr": 1e-7}
    scales = [lr_scale(e, hp, 1e-4, encoder=False) for e in range(20)]
    assert scales[0] == pytest.approx(0.25) and scales[3] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(scales[4:], scales[5:]))  # cosine never climbs
    assert scales[-1] < 0.05


def test_lr_scale_rewarms_the_encoder_after_unfreezing():
    hp = {"warmup_epochs": 2, "epochs": 10, "freeze_encoder_epochs": 2, "min_lr": 1e-7}
    head, encoder = lr_scale(2, hp, 1e-4, encoder=False), lr_scale(2, hp, 1e-4, encoder=True)
    assert encoder == pytest.approx(head / 2)
    assert lr_scale(4, hp, 1e-4, encoder=True) == pytest.approx(lr_scale(4, hp, 1e-4, encoder=False))


def test_build_conformers_marks_failures_none():
    out = build_conformers(["CCO", "not_a_smiles"], 3)
    assert len(out[0]) == 3 and out[1] is None


def test_returned_predictions_use_every_conformer(fitted):
    fold, val_pred, conformers, extra = fitted
    assert np.allclose(fold.predict(conformers[8:], extra[8:]), val_pred, atol=1e-5)


def test_fold_model_round_trips(fitted, tmp_path):
    fold, _, conformers, extra = fitted
    fold.save(str(tmp_path / "fold_0"))
    loaded = FoldModel.load(str(tmp_path / "fold_0"))
    assert np.allclose(loaded.predict(conformers, extra), fold.predict(conformers, extra), atol=1e-6)
