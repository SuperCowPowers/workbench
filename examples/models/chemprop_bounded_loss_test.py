"""Bounded-loss sanity check: verify ChemProp's bounded loss path actually fires.

Builds a small synthetic FeatureSet from public LogP data with artificial
right-censoring, then trains two ChemProp models against the same FeatureSet:

    1. logp-bounded-off  -- hyperparameters omitted (bounded_loss defaults to False)
    2. logp-bounded-on   -- hyperparameters={"bounded_loss": True}

If the bounded loss machinery is working, predictions on the *artificially
censored* rows should look very different between the two models:

    bounded_loss=False:  predictions cluster near the censor value T because the
                         labels are exactly T and MSE/MAE punish predictions on
                         either side of T.
    bounded_loss=True:   predictions are free to go above T without penalty, and
                         since the uncensored half of the high-LogP chemistry is
                         still present at its true (>T) value, the model can
                         learn that high-LogP chemistry actually exists.

To make the difference visible:
- Use a real LogP dataset (chemistry signal the model can actually learn).
- Pick T at a percentile that leaves a meaningful population above the cap.
- Censor ONLY half of the rows above T -- the other half stays at its true
  value so the bounded model has uncensored examples to learn from. Without
  any uncensored examples above T, even bounded_loss=True can't pull
  predictions above T (no information to push them there).

After training, compare predictions on the originally-censored rows in the
two endpoints' cross-fold inference dataframes -- bounded-on should show
predictions distributed above T, bounded-off should show them piled at T.
"""

import numpy as np

from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType, PublicData
from workbench.core.transforms.pandas_transforms.pandas_to_features import PandasToFeatures

# Flip to True to rebuild artifacts that already exist
recreate = False

FS_NAME = "bounded_loss_test_fs"
PUBLIC_KEY = "comp_chem/logp/logp_all"

# Tunables -- these define how visible the bounded-loss effect will be.
N_ROWS = 3000  # subsample size (keeps training fast)
CENSOR_THRESHOLD = 3.0  # T: cap above which rows can be censored (top ~25% of LogP)
CENSOR_FRACTION = 0.5  # fraction of >T rows to actually censor
RANDOM_SEED = 42


def _build_featureset() -> str:
    """Build a synthetic FeatureSet with logp + logp_gt censoring annotation."""
    if not recreate and FeatureSet(FS_NAME).exists():
        return FS_NAME

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
    return FS_NAME


def _train_variant(model_name: str, bounded_loss: bool, description: str) -> None:
    """Train one variant of the model and stand up an endpoint."""
    tags = ["chemprop", "bounded-loss", "synthetic", "logp", "on" if bounded_loss else "off"]

    if recreate or not Model(model_name).exists():
        fs = FeatureSet(FS_NAME)
        kwargs = {}
        if bounded_loss:
            kwargs["hyperparameters"] = {"bounded_loss": True}
        m = fs.to_model(
            name=model_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.CHEMPROP,
            target_column="logp",
            feature_list=["smiles"],
            description=description,
            tags=tags,
            **kwargs,
        )
        m.set_owner("BW")

    if recreate or not Endpoint(model_name).exists():
        end = Model(model_name).to_endpoint(tags=tags)
        end.set_owner("BW")
        end.auto_inference()
        end.cross_fold_inference()


# =============================================================================
# Build FeatureSet, then train two model variants on it
# =============================================================================
_build_featureset()

_train_variant(
    "logp-bounded-off",
    bounded_loss=False,
    description=(
        "Bounded-loss sanity check (control): standard MAE on artificially censored LogP. "
        f"Censored rows have logp = {CENSOR_THRESHOLD} and logp_gt = True; trained without "
        "bounded_loss hyperparameter so the censor labels are treated as exact."
    ),
)

_train_variant(
    "logp-bounded-on",
    bounded_loss=True,
    description=(
        "Bounded-loss sanity check (treatment): BoundedMAE on artificially censored LogP. "
        f"Censored rows have logp = {CENSOR_THRESHOLD} and logp_gt = True; trained with "
        "bounded_loss=True so the censor labels are treated as lower bounds."
    ),
)

print(
    "\nDone. To verify the bounded-loss effect, compare cross-fold predictions "
    "on the censored rows between the two endpoints. NOTE: workbench stores "
    "boolean columns as nullable Int64, so cast logp_gt to bool before "
    "boolean indexing (or compare with == 1):\n\n"
    "  fs   = FeatureSet('bounded_loss_test_fs').pull_dataframe()\n"
    "  fs['logp_gt'] = fs['logp_gt'].fillna(0).astype(bool)\n"
    "  off  = Model('logp-bounded-off').get_inference_predictions('full_cross_fold')\n"
    "  on   = Model('logp-bounded-on' ).get_inference_predictions('full_cross_fold')\n"
    "  off_m = off.merge(fs[['id', 'logp_gt', 'logp_true']], on='id')\n"
    "  on_m  = on.merge(fs[['id', 'logp_gt', 'logp_true']], on='id')\n"
    "  print('off censored preds:', off_m[off_m['logp_gt']]['prediction'].describe())\n"
    "  print('on  censored preds:', on_m [on_m ['logp_gt']]['prediction'].describe())"
)
