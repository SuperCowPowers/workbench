"""Tests for the ChemProp model template: the generated script trains and serves locally."""

import importlib.util
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

# Workbench Imports
from workbench.api import ModelFramework, ModelType
from workbench.model_scripts.script_generation import generate_model_script

pytestmark = pytest.mark.long

SMILES = [
    "CCO",
    "CCN",
    "CCCC",
    "CCOCC",
    "CC(C)O",
    "CC(=O)O",
    "c1ccccc1",
    "Cc1ccccc1",
    "Oc1ccccc1",
    "Nc1ccccc1",
    "Clc1ccccc1",
    "c1ccncc1",
    "C1CCCCC1",
    "C1CCNCC1",
    "CC(N)=O",
    "CCC(=O)O",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "CC(=O)NC1=CC=C(C=C1)O",
]


def _train_and_load(tmp_path, monkeypatch, model_type):
    """Train a tiny ChemProp model from the generated script and load its serving module."""
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    script = generate_model_script(
        {
            "model_type": model_type,
            "model_framework": ModelFramework.CHEMPROP,
            "model_class": None,
            "model_imports": None,
            "target_column": ["target"],
            "feature_list": ["smiles"],
            "compressed_features": [],
            "model_metrics_path": str(metrics),
            "id_column": "id",
            "hyperparameters": {
                "n_folds": 2,
                "max_epochs": 2,
                "patience": 1,
                "hidden_dim": 100,
                "depth": 2,
                "ffn_hidden_dim": 50,
                "batch_size": 16,
                "split_strategy": "random",
            },
        }
    )

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"smiles": SMILES * 4})
    df.insert(0, "id", [f"m{i}" for i in range(len(df))])
    if model_type == ModelType.CLASSIFIER:
        df["target"] = rng.choice(["low", "high"], size=len(df))
    else:
        df["target"] = rng.normal(size=len(df))
    train_dir, model_dir, out_dir = tmp_path / "train", tmp_path / "model", tmp_path / "out"
    for d in (train_dir, model_dir, out_dir):
        d.mkdir()
    df.to_csv(train_dir / "train.csv", index=False)

    cmd = [sys.executable, script, "--train", str(train_dir), "--model-dir", str(model_dir)]
    result = subprocess.run(cmd + ["--output-data-dir", str(out_dir)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]

    monkeypatch.setenv("SM_MODEL_DIR", str(model_dir))
    spec = importlib.util.spec_from_file_location("generated_chemprop_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.model_fn(str(model_dir))


@pytest.mark.parametrize("model_type", [ModelType.UQ_REGRESSOR, ModelType.CLASSIFIER])
def test_all_invalid_batch_keeps_the_output_columns(tmp_path, monkeypatch, model_type):
    """A batch with no parseable SMILES returns the same columns as a mixed batch, all NaN."""
    pytest.importorskip("chemprop")
    module, model_dict = _train_and_load(tmp_path, monkeypatch, model_type)

    mixed = pd.DataFrame({"id": ["a", "b", "c"], "smiles": ["CCO", "not_a_smiles", "c1ccccc1"]})
    invalid = pd.DataFrame({"id": ["x", "y"], "smiles": ["not_a_smiles", ""]})
    mixed_out = module.predict_fn(mixed, model_dict)
    invalid_out = module.predict_fn(invalid, model_dict)

    assert "prediction" in mixed_out.columns
    assert list(invalid_out.columns) == list(mixed_out.columns)
    assert invalid_out["prediction"].isna().all()
    assert mixed_out["prediction"].notna().tolist() == [True, False, True]
