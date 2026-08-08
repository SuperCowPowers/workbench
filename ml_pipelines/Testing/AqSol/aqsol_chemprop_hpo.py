"""Create the AQSol Chemprop regression Model and Endpoint via a hyperparameter search.

Consumes the `fs:aqsol_features` FeatureSet produced by aqsol_feature_set.py.

The `hpo` block runs the search *inside* the single training job — trials are ephemeral,
so no throwaway Workbench models or endpoints are created — and publishes only the winning
config. These hyperparameters run as one of the trials, so the record carries the value an
untuned model would have scored on the same folds.

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
            # The search budget is what the job costs, so it is worth stating. Everything
            # else defaults: https://supercowpowers.github.io/workbench/models/hpo/
            "hpo": {"n_trials": 60, "search_space": space.to_dict()},
        },
    )
    m.set_owner("test")

    # Deploy the Endpoint for the tuned model
    m = Model("aqsol-chemprop-hpo")
    end = m.to_endpoint(tags=["aqsol", "chemprop", "hpo"])
    end.set_owner("test")
    end.test_inference()
    end.cross_fold_inference()
