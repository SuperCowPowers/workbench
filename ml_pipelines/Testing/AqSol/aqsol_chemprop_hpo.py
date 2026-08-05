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
from workbench.training.hpo_harness import SearchSpace

if __name__ == "__main__":

    # The knobs to search — Chemprop's shipped space, unchanged:
    #
    #   depth           IntRange(2, 6, step=1, default=5)
    #   hidden_dim      IntRange(100, 2400, step=100, default=700)
    #   ffn_num_layers  IntRange(1, 3, step=1, default=2)
    #   ffn_hidden_dim  Choice([300, 600, 1800, "300-100", "512-128", "512-128-32", "1024-256-64"])
    #   max_lr          FloatRange(1e-4, 5e-3, log=True, default=1e-3)
    #   batch_size      Choice([64, 128, 256, 512], default=64)
    #
    # It is a dict, so narrowing a range is `space["depth"] = IntRange(3, 5)` and dropping a
    # knob is `del space["ffn_num_layers"]`. space.to_frame() reads back what will be sampled.
    # IntRange / FloatRange / Choice come from workbench.training.hpo_harness.
    space = SearchSpace("chemprop")

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
            # The search budget is what the job costs, so it is worth stating. Everything
            # else defaults: https://supercowpowers.github.io/workbench/models/hpo/
            "hpo": {"n_trials": 40, "search_space": space.to_dict()},
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
