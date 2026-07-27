"""Create the AQSol XGBoost Model and Endpoint via a hyperparameter search over a *custom*
search space.

Same FeatureSet and feature list as aqsol_xgb_hpo.py, which searches the shipped space —
so the pair is a direct A/B on what narrowing the ranges buys.

`SearchSpace` is the editable form of what `model.hpo_search_space()` shows: a dict subclass
keyed by knob, so `[]` and `del` are the whole editing API. `to_dict()` renders it as JSON,
which is what crosses into the training job.

The space here is deliberately opinionated rather than tuned: shallow trees on a 17-feature
tabular set, a learning rate bracketing the XGBoost default, and no `gamma`/`colsample`
because a search that ranks them last is not worth the trials on this data.

Models:
    - aqsol-xgb-custom-space
Endpoints:
    - aqsol-xgb-custom-space
"""

from workbench.api import FeatureSet, Model, ModelFramework, ModelType
from workbench.training.hpo_harness import FloatRange, IntRange, SearchSpace

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


def custom_space() -> dict:
    """Narrow the shipped XGBoost space, then hand back the JSON the `hpo` block takes."""
    space = SearchSpace("xgboost")

    # Tighter than the shipped 3-16 / 0.003-0.3: 17 tabular features don't need deep trees,
    # and the default learning rate (0.05) sits mid-range here rather than at an edge.
    space["max_depth"] = IntRange(4, 8, default=6)
    space["learning_rate"] = FloatRange(0.02, 0.12, log=True, default=0.05)

    # Drop the knobs this dataset doesn't reward; every knob costs trials.
    del space["gamma"]
    del space["colsample_bytree"]

    return space.to_dict()


if __name__ == "__main__":

    # Build the searched XGBoost Model from the aqsol_features FeatureSet
    fs = FeatureSet("aqsol_features")
    m = fs.to_model(
        name="aqsol-xgb-custom-space",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="solubility",
        feature_list=FEATURE_LIST,
        description="AQSol XGBoost regressor (custom search space)",
        tags=["aqsol", "xgboost", "hpo"],
        hyperparameters={
            "uq_version": "v1",
            # A knob set here is the baseline the search must beat, and the value for any
            # knob the space leaves out — gamma and colsample_bytree train at these.
            "gamma": 0.0,
            "colsample_bytree": 0.8,
            "hpo": {
                "backend": "optuna",
                "n_trials": 100,
                "search_space": custom_space(),
                "rerank_top_k": 5,
            },
        },
    )
    m.set_owner("test")

    # Deploy the Endpoint for the searched model
    m = Model("aqsol-xgb-custom-space")
    end = m.to_endpoint(tags=["aqsol", "xgboost", "hpo"])
    end.set_owner("test")
    end.test_inference()
    end.cross_fold_inference()
