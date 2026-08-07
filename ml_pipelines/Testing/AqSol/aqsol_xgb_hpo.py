"""Create the AQSol XGBoost regression Model and Endpoint via a hyperparameter search.

Consumes the `fs:aqsol_features` FeatureSet produced by aqsol_feature_set.py, on the same
feature list as aqsol_reg_1.py so the tuned model is a direct A/B against the untuned one.

The `hpo` block runs the search *inside* the single training job — trials are ephemeral,
so no throwaway Workbench models or endpoints are created — and publishes only the winning
config. The search shortlists; a re-rank then scores those finalists and these
hyperparameters as-is, and whichever wins there is published. A search that finds nothing
real therefore publishes the untuned baseline.

An XGBoost trial is seconds, not minutes, so this is the cheap way to exercise the search
machinery end to end.

Models:
    - aqsol-xgb-hpo
Endpoints:
    - aqsol-xgb-hpo
"""

from workbench.api import FeatureSet, Model, ModelFramework, ModelType

FEATURE_LIST = [
    "molwt",
    "mollogp",
    "molmr",
    "heavyatomcount",
    "numhacceptors",
    "numhdonors",
    "numheteroatoms",
    "numrotatablebonds",
    "numvalenceelectrons",
    "numaromaticrings",
    "numsaturatedrings",
    "numaliphaticrings",
    "ringcount",
    "tpsa",
    "labuteasa",
    "balabanj",
    "bertzct",
]

if __name__ == "__main__":

    # Build the hyperparameter-searched XGBoost Model from the aqsol_features FeatureSet
    fs = FeatureSet("aqsol_features")
    m = fs.to_model(
        name="aqsol-xgb-hpo",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="solubility",
        feature_list=FEATURE_LIST,
        description="AQSol XGBoost regressor (hyperparameter-searched)",
        tags=["aqsol", "xgboost", "hpo"],
        hyperparameters={
            "hpo": {
                # The base training image carries optuna, not ray — and one XGBoost fit
                # already spreads across every core, so the search is serial by design.
                "backend": "optuna",
                "n_trials": 250,
            },
        },
        # For an out-of-distribution objective, pass validation_ids=[...] and set
        # hpo["metric"]="holdout_mae"; those rows are held out of training either way.
    )
    m.set_owner("test")

    # Deploy the Endpoint for the tuned model
    m = Model("aqsol-xgb-hpo")
    end = m.to_endpoint(tags=["aqsol", "xgboost", "hpo"])
    end.set_owner("test")
    end.test_inference()
    end.cross_fold_inference()
