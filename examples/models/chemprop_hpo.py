"""Hyperparameter-search example: tune a Chemprop regressor on the AQSol feature set.

The ``hpo`` hyperparameter block runs a hyperparameter SEARCH *inside* the single
training job — the trials are ephemeral (no throwaway Workbench models/endpoints) —
and only the winning config is published as the model. Search knobs and the objective
live in ``workbench.training.chemprop_hpo``.
"""

from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType
from workbench.training.hpo_harness import SearchSpace

# Recreate flag in case you want to recreate the artifacts
recreate = True
model_name = "aqsol-chemprop-hpo"

# The knobs to search — Chemprop's shipped space, unchanged:
#
#   depth           IntRange(2, 6, step=1, default=5)
#   hidden_dim      IntRange(100, 2400, step=100, default=700)
#   ffn_hidden_dim  Choice(["300", "600", "1200", "1800", "300-300", "300-100", ...])
#   max_lr          FloatRange(1e-4, 5e-3, log=True, default=1e-3)
#   batch_size      Choice([64, 128, 256, 512], default=64)
#
# ffn_hidden_dim is a per-layer shape: its length is the head's depth, so it covers what
# chemprop splits across ffn_hidden_dim + ffn_num_layers, plus the tapered heads.
#
# It is a dict, so narrowing a range is `space["depth"] = IntRange(3, 5)` and dropping a knob
# is `del space["depth"]`. space.to_frame() reads back what will be sampled.
# IntRange / FloatRange / Choice come from workbench.training.hpo_harness.
space = SearchSpace("chemprop")

# =============================================================================
# Hyperparameter-searched Chemprop Regression Model
# =============================================================================
if recreate or not Model(model_name).exists():
    fs = FeatureSet("aqsol_features")
    m = fs.to_model(
        name=model_name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="solubility",
        feature_list=["smiles"],
        description="AQSol Chemprop regressor (hyperparameter-searched)",
        tags=["aqsol", "chemprop", "hpo"],
        hyperparameters={
            # The search budget is what the job costs, so it is worth stating. Everything
            # else defaults: https://supercowpowers.github.io/workbench/models/hpo/
            "hpo": {"n_trials": 60, "search_space": space.to_dict()},
        },
        # The objective is `cv_mae` over scaffold folds of the training rows.
    )
    m.set_owner("BW")

# Create an Endpoint for the tuned model
if recreate or not Endpoint(model_name).exists():
    end = Model(model_name).to_endpoint(tags=["aqsol", "chemprop", "hpo"])
    end.set_owner("BW")
    end.test_inference()
    end.cross_fold_inference()
