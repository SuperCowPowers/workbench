"""Tests for the TabICL model template: the generated script trains and serves locally."""

import ast
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

# Workbench Imports
from workbench.api import ModelFramework, ModelType
from workbench.model_scripts.script_generation import generate_model_script

FEATURES = [f"f{i}" for i in range(8)]


def _generate(tmp_path, model_type=ModelType.UQ_REGRESSOR):
    metrics = tmp_path / "metrics"
    metrics.mkdir(exist_ok=True)
    return generate_model_script(
        {
            "model_type": model_type,
            "model_framework": ModelFramework.TABICL,
            "model_class": None,
            "model_imports": None,
            "target_column": "target",
            "feature_list": FEATURES,
            "compressed_features": [],
            "model_metrics_path": str(metrics),
            "id_column": "id",
            "hyperparameters": {"n_estimators": 2, "n_folds": 2, "shap_sample_size": 3},
        }
    )


def _training_csv(tmp_path, sample_weight=None):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(240, len(FEATURES))), columns=FEATURES)
    df["target"] = df["f0"] * 2 + df["f1"] - df["f2"] + rng.normal(scale=0.3, size=len(df))
    df.insert(0, "id", [f"m{i}" for i in range(len(df))])
    df["validation"] = [i >= 200 for i in range(len(df))]
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


def test_tabicl_framework_selects_a_valid_script(tmp_path):
    source = open(_generate(tmp_path)).read()
    ast.parse(source)
    assert "TabICL Model Template" in source


@pytest.mark.medium
def test_classification_is_refused(tmp_path):
    pytest.importorskip("tabicl")
    _, train_dir = _training_csv(tmp_path)
    result, _ = _train(_generate(tmp_path, ModelType.CLASSIFIER), tmp_path, train_dir)
    assert result.returncode != 0 and "regression only" in result.stderr


@pytest.mark.medium
def test_non_uniform_sample_weights_are_refused(tmp_path):
    pytest.importorskip("tabicl")
    _, train_dir = _training_csv(tmp_path, sample_weight=[1.0] * 239 + [0.5])
    result, _ = _train(_generate(tmp_path), tmp_path, train_dir)
    assert result.returncode != 0 and "does not support sample weights" in result.stderr


@pytest.mark.medium
def test_generated_script_trains_and_serves(tmp_path, monkeypatch):
    pytest.importorskip("tabicl.shap")
    df, train_dir = _training_csv(tmp_path, sample_weight=1.0)
    script = _generate(tmp_path)
    result, model_dir = _train(script, tmp_path, train_dir)
    assert result.returncode == 0, result.stderr[-2000:]

    # Training outputs: held-out rows scored, SHAP ranks the real drivers, memory probed
    metrics = tmp_path / "metrics"
    oof, val = pd.read_csv(metrics / "oof_predictions.csv"), pd.read_csv(metrics / "val_predictions.csv")
    assert len(oof) == 200 and len(val) == 40
    assert {"prediction", "prediction_std", "confidence", "q_025", "q_975"} <= set(val.columns)
    importance = json.load(open(metrics / "shap_importance.json"))
    assert {name for name, _ in importance[:3]} == {"f0", "f1", "f2"}
    assert json.load(open(metrics / "inference_profile.json"))["inference_memory_gb"] > 0

    # Serving path: the endpoint's model_fn + predict_fn from the same script
    monkeypatch.setenv("SM_MODEL_DIR", str(model_dir))
    spec = importlib.util.spec_from_file_location("generated_tabicl_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = module.predict_fn(df[["id"] + FEATURES].tail(10).reset_index(drop=True), module.model_fn(str(model_dir)))
    assert len(out) == 10
    assert {"prediction", "prediction_std", "confidence", "q_025", "q_975"} <= set(out.columns)
    assert (out["prediction_std"] > 0).all()
