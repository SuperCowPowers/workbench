"""Hyperparameter-search example: tune a Chemprop regressor on the AQSol feature set.

The ``hpo`` hyperparameter block runs a hyperparameter SEARCH *inside* the single
training job — the trials are ephemeral (no throwaway Workbench models/endpoints) —
and only the winning config is published as the model. Search knobs and the objective
live in ``workbench.training.chemprop_hpo``.
"""

from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType

# Recreate flag in case you want to recreate the artifacts
recreate = True
model_name = "aqsol-chemprop-hpo"

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
            "uq_version": "v1",
            "hpo": {
                # "ray" + max_parallel > 1 runs trials concurrently (ASHA) and puts the job
                # on a 4-GPU instance; "optuna" is the serial single-GPU path.
                "backend": "ray",
                "max_parallel": 8,  # 4 GPUs x 2 trials each (see gpus_per_trial)
                "n_trials": 40,  # 5 baseline trials (pruner warmup) + 35 pruned candidates
                "search_space": "basic",  # "basic" (architecture capacity) | "basic+lr"
            },
        },
        # For an honest out-of-distribution objective, pass validation_ids=[...]: those
        # rows are held out of training and scored as `holdout_mae`. Without them the
        # search optimizes `cv_mae` on a single scaffold split of the training rows.
    )
    m.set_owner("BW")

# Create an Endpoint for the tuned model
if recreate or not Endpoint(model_name).exists():
    end = Model(model_name).to_endpoint(tags=["aqsol", "chemprop", "hpo"])
    end.set_owner("BW")
    end.test_inference()
    end.cross_fold_inference()
