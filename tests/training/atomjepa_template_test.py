"""Tests for the Atom-JEPA model template: the generated script trains and serves locally.

Fine-tuning on CPU is slow, so the data and the schedule are tiny.
"""

import ast
import importlib.util
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

# Workbench Imports
from workbench.api import ModelFramework, ModelType
from workbench.model_scripts.script_generation import generate_model_script

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
    "C1CCNCC1",
    "c1ccoc1",
    "CN1CCOCC1",
    "Oc1ccc(Br)cc1",
]
HYPERPARAMETERS = {
    "n_folds": 2,
    "epochs": 2,
    "batch_size": 4,
    "warmup_epochs": 1,
    "freeze_encoder_epochs": 1,
    "n_conformers": 2,
    "selection_conformers": 1,
}


def _generate(tmp_path, model_type=ModelType.UQ_REGRESSOR, features=("smiles", "x"), compressed=()):
    metrics = tmp_path / "metrics"
    metrics.mkdir(exist_ok=True)
    return generate_model_script(
        {
            "model_type": model_type,
            "model_framework": ModelFramework.ATOMJEPA,
            "model_class": None,
            "model_imports": None,
            "target_column": "target",
            "feature_list": list(features),
            "compressed_features": list(compressed),
            "model_metrics_path": str(metrics),
            "id_column": "id",
            "hyperparameters": HYPERPARAMETERS,
        }
    )


def _training_csv(tmp_path, sample_weight=None):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": [f"m{i}" for i in range(len(SMILES))], "smiles": SMILES})
    df["x"] = rng.normal(size=len(df))
    df["target"] = df["x"] + rng.normal(scale=0.3, size=len(df))
    df["validation"] = [i >= 12 for i in range(len(df))]
    if sample_weight is not None:
        df["sample_weight"] = sample_weight
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    df.to_csv(train_dir / "train.csv", index=False)
    return df, train_dir


def _train(script, tmp_path, train_dir):
    model_dir, out_dir = tmp_path / "model", tmp_path / "out"
    model_dir.mkdir()
    out_dir.mkdir()
    cmd = [sys.executable, script, "--train", str(train_dir), "--model-dir", str(model_dir)]
    result = subprocess.run(cmd + ["--output-data-dir", str(out_dir)], capture_output=True, text=True)
    return result, model_dir


def test_atomjepa_framework_selects_a_valid_script(tmp_path):
    source = open(_generate(tmp_path)).read()
    ast.parse(source)
    assert "Atom-JEPA Model Template" in source


@pytest.mark.medium
def test_classification_is_refused(tmp_path):
    pytest.importorskip("atom_jepa")
    _, train_dir = _training_csv(tmp_path)
    result, _ = _train(_generate(tmp_path, ModelType.CLASSIFIER), tmp_path, train_dir)
    assert result.returncode != 0 and "regression only" in result.stderr


@pytest.mark.medium
def test_non_uniform_sample_weights_are_refused(tmp_path):
    pytest.importorskip("atom_jepa")
    _, train_dir = _training_csv(tmp_path, sample_weight=[1.0] * 15 + [0.5])
    result, _ = _train(_generate(tmp_path), tmp_path, train_dir)
    assert result.returncode != 0 and "does not support sample weights" in result.stderr


@pytest.mark.medium
def test_compressed_features_are_refused(tmp_path):
    pytest.importorskip("atom_jepa")
    _, train_dir = _training_csv(tmp_path)
    result, _ = _train(_generate(tmp_path, features=("smiles", "emb"), compressed=("emb",)), tmp_path, train_dir)
    assert result.returncode != 0 and "not compressed features" in result.stderr


@pytest.mark.medium
def test_generated_script_trains_and_serves(tmp_path, monkeypatch):
    pytest.importorskip("atom_jepa")
    df, train_dir = _training_csv(tmp_path, sample_weight=1.0)
    script = _generate(tmp_path)
    result, model_dir = _train(script, tmp_path, train_dir)
    assert result.returncode == 0, result.stderr[-2000:]

    # Training outputs: out-of-fold and held-out rows, both with the ensemble spread and UQ
    metrics = tmp_path / "metrics"
    oof, val = pd.read_csv(metrics / "oof_predictions.csv"), pd.read_csv(metrics / "val_predictions.csv")
    assert len(oof) == 12 and len(val) == 4
    for frame in (oof, val):
        assert {"prediction", "prediction_std", "confidence", "q_025", "q_975"} <= set(frame.columns)
        assert (frame["prediction_std"] > 0).all()

    # Serving path: the endpoint's model_fn + predict_fn from the same script; a SMILES with no
    # conformer gets NaN rather than failing the batch
    monkeypatch.setenv("SM_MODEL_DIR", str(model_dir))
    spec = importlib.util.spec_from_file_location("generated_atomjepa_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batch = pd.concat([df[["id", "smiles", "x"]].tail(3), pd.DataFrame({"id": ["bad"], "smiles": ["xyz"], "x": [0.0]})])
    out = module.predict_fn(batch.reset_index(drop=True), module.model_fn(str(model_dir)))
    assert len(out) == 4
    assert out["prediction"].iloc[:3].notna().all() and np.isnan(out["prediction"].iloc[3])
    assert {"prediction_std", "confidence", "q_025", "q_975"} <= set(out.columns)
