"""Create the AQSol Chemprop regression Model and Endpoint via a hyperparameter search.

Consumes the `fs:aqsol_features` FeatureSet produced by aqsol_feature_set.py.

The `hpo` block runs the search *inside* the single training job — trials are ephemeral,
so no throwaway Workbench models or endpoints are created — and publishes only the winning
config. The search shortlists; a re-rank then scores those finalists and these
hyperparameters as-is, and whichever wins there is published. A search that finds nothing
real therefore publishes the untuned baseline.

Heavy: a multi-GPU box for hours. Launch it on AWS Batch rather than inline.

Models:
    - aqsol-chemprop-hpo
Endpoints:
    - aqsol-chemprop-hpo
"""

from workbench.api import FeatureSet, Model, ModelFramework, ModelType

if __name__ == "__main__":

    # Build the hyperparameter-searched Chemprop Model from the aqsol_features FeatureSet
    fs = FeatureSet("aqsol_features")
    m = fs.to_model(
        name="aqsol-chemprop-hpo",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="solubility",
        feature_list=["smiles"],
        description="AQSol Chemprop regressor (hyperparameter-searched)",
        tags=["aqsol", "chemprop", "hpo"],
        hyperparameters={
            "uq_version": "v1",
            "hpo": {
                # "ray" + max_parallel > 1 runs trials concurrently (ASHA) on a 4-GPU instance.
                "backend": "ray",
                "max_parallel": 8,  # 4 GPUs x 2 trials each
                "n_trials": 60,
                # search_space defaults to "basic+lr" (capacity + LR schedule + batch size).
                "rerank_top_k": 5,
            },
        },
        # For an out-of-distribution objective, pass validation_ids=[...] and set
        # hpo["metric"]="holdout_mae"; those rows are held out of training either way.
    )
    m.set_owner("test")

    # Deploy the Endpoint for the tuned model
    m = Model("aqsol-chemprop-hpo")
    end = m.to_endpoint(tags=["aqsol", "chemprop", "hpo"])
    end.set_owner("test")
    end.test_inference()
    end.cross_fold_inference()
