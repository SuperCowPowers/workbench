"""Open ADMET mppb: hyperparameter-searched Chemprop, alongside the hand-tuned one.

Same FeatureSet, target and held-out capture as the `mppb-reg-chemprop` model in mppb.py —
only the knobs are searched rather than defaulted. The search runs inside the single
training job; trials are ephemeral, so only the winning config is published.

mppb is the small-data half of the pair with ../logd/logd_chemprop_hpo.py:

  - 1302 rows against logd's 5039 — a quarter the data, and a scaffold fold holds ~260
  - std 0.46 over a 0.0 to 1.95 range — much less signal per compound than logd
  - the size where most real project datasets sit, and where the literature has chemprop
    HPO at roughly a coin flip against defaults (Tetko et al., J Cheminform 2024)

Running both answers the question one of them alone cannot: whether an HPO margin tracks
dataset size, or whether it is noise at both ends.

Reading the result: the search's own margin is optimistic (it is a minimum over many noisy
trials, scored on the folds that selected it), and the smaller the assay the more optimistic
it gets. The honest number is the `rx_test_mppb` capture — molecules that are not in the
FeatureSet at all — compared against the same capture from the untuned `mppb-reg-chemprop`.

Build the FeatureSet first: python ../load_data/load_data.py
"""

from workbench.api import DFStore, FeatureSet, ModelFramework, ModelType

FS_NAME = "open_admet_mppb"
TARGET = "mppb"
MODEL_NAME = "mppb-reg-chemprop-hpo"
TAGS = ["open_admet", TARGET, "regression", "chemprop", "hpo"]

# Featurized + log-transformed ExpansionRx test set, written by ../load_data/load_data.py
TEST_STORE_KEY = "/workbench/datasets/open_admet_rx_test_featurized"


def main():
    fs = FeatureSet(FS_NAME)

    model = fs.to_model(
        name=MODEL_NAME,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column=TARGET,
        feature_list=["smiles"],
        description=f"ChemProp D-MPNN for {TARGET} (hyperparameter-searched)",
        tags=TAGS,
        # Everything defaults: the shipped chemprop search space, 60 trials, pooled
        # out-of-fold MAE on scaffold folds. https://supercowpowers.github.io/workbench/models/hpo/
        hyperparameters={"hpo": {}},
    )
    model.set_owner("BW")

    end = model.to_endpoint(tags=TAGS, max_concurrency=1)
    end.set_owner("BW")
    end.test_inference()
    end.cross_fold_inference()

    # The held-out ExpansionRx rows: never in the FeatureSet, so never trained on, validated
    # against, or used to calibrate UQ. This capture is what settles whether HPO helped.
    test_df = DFStore().get(TEST_STORE_KEY).dropna(subset=[TARGET])
    print(f"Running held-out test inference for {end.name} on {len(test_df)} rows")
    end.inference(test_df[["molecule_name", "smiles", TARGET]], capture_name=f"rx_test_{TARGET}")


if __name__ == "__main__":
    main()
