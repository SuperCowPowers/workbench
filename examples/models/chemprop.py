"""Baseline example: a plain 5-fold Chemprop regressor on the AQSol feature set.

The hyperparameter-searched counterpart is ``chemprop_hpo.py`` — this is the same
model and feature set with the ``hpo`` block dropped, so the two are directly
comparable (the search should match or beat this baseline). Open-ADMET multi-task /
hybrid / classifier examples live in ``chemprop_open_admet.py``.
"""

from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType

# Recreate flag in case you want to recreate the artifacts
recreate = True
model_name = "aqsol-chemprop"

# =============================================================================
# Chemprop Regression Model (baseline for the HPO comparison)
# =============================================================================
if recreate or not Model(model_name).exists():
    fs = FeatureSet("aqsol_features")
    m = fs.to_model(
        name=model_name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="solubility",
        feature_list=["smiles"],
        description="AQSol Chemprop regressor (hand-tuned baseline)",
        tags=["aqsol", "chemprop"],
        hyperparameters={"uq_version": "v1"},
    )
    m.set_owner("BW")

# Create an Endpoint for the model
if recreate or not Endpoint(model_name).exists():
    end = Model(model_name).to_endpoint(tags=["aqsol", "chemprop"])
    end.set_owner("BW")
    end.test_inference()
    end.cross_fold_inference()
