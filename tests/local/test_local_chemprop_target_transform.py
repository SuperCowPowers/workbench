"""A multi-target ChemProp model with a target transform, through the real generated script.

ChemProp is the one template that reports per target, so this covers the per-target
forward on entry and ``report_targets`` on exit, for out-of-fold, held-out and endpoint rows.
"""

import numpy as np
import pandas as pd
import pytest

from workbench.local import DataSource, ModelFramework, ModelType
from workbench.utils.config_manager import ConfigManager

pytestmark = pytest.mark.long

FRAGMENTS = ["C", "CC", "CCC", "CCCC", "CO", "CCO", "CCN", "CN", "c1ccccc1", "c1ccncc1", "CC(C)", "CS", "CCl", "CF"]


@pytest.fixture(autouse=True)
def local_storage(tmp_path):
    """Point local storage at a temp directory for every test"""
    cm = ConfigManager()
    original = cm.config.get("WORKBENCH_LOCAL_PATH")
    cm.set_config("WORKBENCH_LOCAL_PATH", str(tmp_path))
    yield tmp_path
    cm.set_config("WORKBENCH_LOCAL_PATH", original)


@pytest.fixture
def molecule_feature_set():
    """Small molecules with a target spanning decades (cmax) and a linear one (logd)."""
    rng = np.random.default_rng(3)
    smiles = sorted({a + b for a in FRAGMENTS for b in FRAGMENTS})[:120]
    df = pd.DataFrame({"id": range(len(smiles)), "smiles": smiles})
    size = df["smiles"].str.len().to_numpy(dtype=float)
    df["cmax"] = np.power(10.0, 0.3 * size + rng.normal(scale=0.1, size=len(df)))
    df["logd"] = 0.2 * size + rng.normal(scale=0.1, size=len(df))
    return DataSource(df, name="molecule_data").to_features("molecule_features", id_column="id")


def test_multi_target_log10_reports_each_target_on_its_own_scale(molecule_feature_set):
    truth = molecule_feature_set.pull_dataframe().set_index("id")
    held_out = list(range(100, 120))
    model = molecule_feature_set.to_model(
        "chemprop-log10-model",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column=["cmax", "logd"],
        feature_list=["smiles"],
        hyperparameters={
            "target_transform": {"cmax": {"name": "log10"}},
            "prediction_bounds": {"cmax": [0.0, 1.0e6]},
            "n_folds": 2,
            "max_epochs": 15,
            "hidden_dim": 64,
            "depth": 2,
        },
        validation_ids=held_out,
    )
    assert model.training_state()["returncode"] == 0, model.training_log()

    for frame in (model.oof_predictions().set_index("id"), model.validation_predictions().set_index("id")):
        # Truth comes back on the original scale, whether it was forwarded for training or not
        np.testing.assert_allclose(frame["cmax"], truth.loc[frame.index, "cmax"], rtol=1e-4)
        np.testing.assert_allclose(frame["logd"], truth.loc[frame.index, "logd"], rtol=1e-4)

        # The transformed target reports in original units, with its spread in log units
        assert frame["cmax_pred"].min() > 0.0 and frame["cmax_pred"].max() > 20
        assert frame["cmax_pred_std"].max() < 5.0
        np.testing.assert_allclose(frame["prediction"], frame["cmax_pred"])

        # The primary's unprefixed quantiles are inverted and clipped to its bounds
        quantiles = [c for c in frame.columns if c.startswith("q_")]
        assert quantiles and frame[quantiles].min().min() > 0.0
        assert frame[quantiles].max().max() <= 1.0e6

        # The untransformed target is untouched by its neighbour's transform
        assert frame["logd_pred"].abs().max() < 20

    # Out-of-fold rows carry per-target quantiles too, reported like the aliases
    oof = model.oof_predictions()
    np.testing.assert_allclose(oof["cmax_q_95"], oof["q_95"])
    assert len(model.validation_predictions()) > 0

    predictions = model.to_endpoint().inference(molecule_feature_set.pull_dataframe())
    assert predictions["cmax_pred"].min() > 0.0 and predictions["cmax_pred"].max() > 20
    assert predictions["cmax_pred_std"].max() < 5.0
