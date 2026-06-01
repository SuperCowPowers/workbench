"""Unit tests for model script generation."""

from enum import Enum
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


class ModelType(Enum):
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    UQ_REGRESSOR = "uq_regressor"
    ENSEMBLE_REGRESSOR = "ensemble_regressor"


class ModelFramework(Enum):
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    PYTORCH = "pytorch"
    CHEMPROP = "chemprop"
    META = "meta"


def load_script_generation(monkeypatch):
    api_module = ModuleType("workbench.api")
    api_module.ModelType = ModelType
    api_module.ModelFramework = ModelFramework
    monkeypatch.setitem(sys.modules, "workbench.api", api_module)

    module_path = Path(__file__).resolve().parents[2] / "src" / "workbench" / "model_scripts" / "script_generation.py"
    spec = importlib.util.spec_from_file_location("script_generation_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_template_params(model_type):
    return {
        "model_type": model_type,
        "model_framework": ModelFramework.LIGHTGBM,
        "id_column": "id",
        "target_column": "target",
        "feature_list": ["a", "b"],
        "compressed_features": [],
        "model_metrics_s3_path": "s3://bucket/models/lightgbm",
        "train_all_data": True,
        "hyperparameters": {"n_estimators": 12},
        "model_class": None,
        "model_imports": None,
    }


def test_lightgbm_regressor_script_generation(monkeypatch):
    script_generation = load_script_generation(monkeypatch)

    script_path = Path(script_generation.generate_model_script(base_template_params(ModelType.REGRESSOR)))
    script = script_path.read_text()

    assert "from lightgbm import LGBMRegressor" in script
    assert '"model_class": LGBMRegressor' in script
    assert "\"hyperparameters\": {'n_estimators': 12}" in script
    assert (script_path.parent / "requirements.txt").read_text().strip() == "lightgbm>=4.6.0"


def test_lightgbm_classifier_script_generation(monkeypatch):
    script_generation = load_script_generation(monkeypatch)

    script_path = Path(script_generation.generate_model_script(base_template_params(ModelType.CLASSIFIER)))
    script = script_path.read_text()

    assert "from lightgbm import LGBMClassifier" in script
    assert '"model_class": LGBMClassifier' in script


def test_lightgbm_rejects_unsupported_model_type(monkeypatch):
    script_generation = load_script_generation(monkeypatch)

    with pytest.raises(ValueError, match="LightGBM model generation does not support"):
        script_generation.generate_model_script(base_template_params(ModelType.UQ_REGRESSOR))
