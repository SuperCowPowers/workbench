"""Bounded-Loss Sanity Check (ChemProp)

This script verifies that ChemProp's bounded-loss path (BoundedMAE / BoundedMSE)
fires correctly when a FeatureSet provides per-target `_gt` / `_lt` censor
columns and the model is trained with hyperparameters={"bounded_loss": True}.

It builds a small synthetic FeatureSet from public LogP data with artificial
right-censoring, then trains two ChemProp variants against the same FeatureSet:

    1. logp-bounded-off  -- hyperparameters omitted (bounded_loss defaults to False)
    2. logp-bounded-on   -- hyperparameters={"bounded_loss": True}

Predictions on the *artificially censored* rows should look very different:

    bounded_loss=False:  predictions cluster near the censor value T because the
                         labels are exactly T and MAE punishes deviations in
                         either direction.
    bounded_loss=True:   predictions are free to go above T without penalty, and
                         since the uncensored half of the high-LogP chemistry
                         remains at its true (>T) value, the model can learn
                         that high-LogP chemistry actually exists.

Test design notes
-----------------
- Use a real LogP dataset so the model has actual chemistry signal to learn.
- Pick T at a percentile that leaves a meaningful population above the cap.
- Censor ONLY half of the rows above T -- the other half stays at its true
  value so the bounded model has uncensored examples to learn from. Without
  any uncensored examples above T, even bounded_loss=True can't pull
  predictions above T (no information to push them there).

Verifying the bounded-loss effect after training
------------------------------------------------
Note: workbench stores boolean FS columns as nullable Int64, so cast logp_gt to
bool before boolean indexing (or compare with == 1):

    fs = FeatureSet("bounded_loss_test_fs").pull_dataframe()
    fs["logp_gt"] = fs["logp_gt"].fillna(0).astype(bool)
    off = Model("logp-bounded-off").get_inference_predictions("full_cross_fold")
    on  = Model("logp-bounded-on" ).get_inference_predictions("full_cross_fold")
    off_m = off.merge(fs[["id", "logp_gt", "logp_true"]], on="id")
    on_m  = on .merge(fs[["id", "logp_gt", "logp_true"]], on="id")
    for name, m in [("off", off_m), ("on", on_m)]:
        c = m[m["logp_gt"]]
        err = (c["prediction"] - c["logp_true"]).abs()
        print(f"{name}: MAE on censored vs true = {err.mean():.3f}")
"""

import numpy as np

from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType, PublicData
from workbench.core.transforms.pandas_transforms.pandas_to_features import PandasToFeatures

# Recreate Flag in case you want to recreate the artifacts
recreate = False

# Synthetic dataset tunables -- these define how visible the bounded-loss effect is
FS_NAME = "bounded_loss_test_fs"
PUBLIC_KEY = "comp_chem/logp/logp_all"
N_ROWS = 3000  # subsample size (keeps training fast)
CENSOR_THRESHOLD = 3.0  # T: cap above which rows can be censored (top ~25% of LogP)
CENSOR_FRACTION = 0.5  # fraction of >T rows to actually censor
RANDOM_SEED = 42


# =============================================================================
# Build Synthetic FeatureSet (LogP with artificial right-censoring)
# =============================================================================
# Pulls real LogP data from public storage, censors half of the rows above T
# (recorded logp clipped to T, logp_gt flag set), and keeps the other half at
# their true values. logp_true is preserved for post-training analysis.

if recreate or not FeatureSet(FS_NAME).exists():
    df = PublicData().get(PUBLIC_KEY)
    if df is None:
        raise RuntimeError(f"Could not load {PUBLIC_KEY} from public data")

    df = df[["smiles", "logp"]].dropna().reset_index(drop=True)
    df = df.sample(n=min(N_ROWS, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)
    df["id"] = df.index.astype(int)

    # Preserve the true value so we can analyze post-training
    df["logp_true"] = df["logp"]

    # Censor half of the rows with logp > T: set recorded logp = T and mark _gt
    rng = np.random.default_rng(RANDOM_SEED)
    above = df.index[df["logp_true"] > CENSOR_THRESHOLD]
    n_to_censor = int(len(above) * CENSOR_FRACTION)
    censored_idx = rng.choice(above, size=n_to_censor, replace=False)

    df["logp_gt"] = False
    df.loc[censored_idx, "logp_gt"] = True
    df.loc[censored_idx, "logp"] = CENSOR_THRESHOLD  # clip the label to the cap

    df = df[["id", "smiles", "logp", "logp_gt", "logp_true"]]
    n_above = int((df["logp_true"] > CENSOR_THRESHOLD).sum())
    n_censored = int(df["logp_gt"].sum())
    print(
        f"FeatureSet '{FS_NAME}': {len(df)} rows | "
        f"true logp > {CENSOR_THRESHOLD}: {n_above} | censored (logp_gt=True): {n_censored} | "
        f"uncensored above cap: {n_above - n_censored}"
    )

    to_features = PandasToFeatures(FS_NAME)
    to_features.set_input(df, id_column="id")
    to_features.set_output_tags(["chemprop", "bounded-loss", "synthetic", "logp"])
    to_features.transform()


# =============================================================================
# Control Model (bounded_loss=False -- default MAE)
# =============================================================================
# Trained without the bounded_loss hyperparameter so the censor labels are
# treated as exact targets. Predictions on censored rows should cluster near
# the cap value (T=3.0) because MAE punishes deviations in either direction.

if recreate or not Model("logp-bounded-off").exists():
    feature_set = FeatureSet(FS_NAME)
    m = feature_set.to_model(
        name="logp-bounded-off",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="logp",
        feature_list=["smiles"],
        description=(
            "Bounded-loss sanity check (control): standard MAE on artificially censored "
            f"LogP. Censored rows have logp = {CENSOR_THRESHOLD} and logp_gt = True; "
            "trained without bounded_loss hyperparameter so censor labels are treated as exact."
        ),
        tags=["chemprop", "bounded-loss", "synthetic", "logp", "off"],
    )
    m.set_owner("BW")

# Create an Endpoint
if recreate or not Endpoint("logp-bounded-off").exists():
    m = Model("logp-bounded-off")
    end = m.to_endpoint(tags=["chemprop", "bounded-loss", "synthetic", "logp", "off"])
    end.set_owner("BW")
    end.test_inference()
    end.cross_fold_inference()


# =============================================================================
# Treatment Model (bounded_loss=True -- BoundedMAE)
# =============================================================================
# Trained with hyperparameters={"bounded_loss": True} so the logp_gt column is
# read by the template and censor labels become lower bounds. Predictions on
# censored rows should spread above the cap, using the chemistry signal from
# the uncensored half of the high-LogP rows.

if recreate or not Model("logp-bounded-on").exists():
    feature_set = FeatureSet(FS_NAME)
    m = feature_set.to_model(
        name="logp-bounded-on",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="logp",
        feature_list=["smiles"],
        description=(
            "Bounded-loss sanity check (treatment): BoundedMAE on artificially censored "
            f"LogP. Censored rows have logp = {CENSOR_THRESHOLD} and logp_gt = True; "
            "trained with bounded_loss=True so censor labels are treated as lower bounds."
        ),
        tags=["chemprop", "bounded-loss", "synthetic", "logp", "on"],
        hyperparameters={"bounded_loss": True},
    )
    m.set_owner("BW")

# Create an Endpoint
if recreate or not Endpoint("logp-bounded-on").exists():
    m = Model("logp-bounded-on")
    end = m.to_endpoint(tags=["chemprop", "bounded-loss", "synthetic", "logp", "on"])
    end.set_owner("BW")
    end.test_inference()
    end.cross_fold_inference()
