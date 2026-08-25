"""CYP XGBoost UQ models on 2D + curated xTB 3D-v2 features — one per isoform.

XGBoost is single-target, so each of the four scored isoforms gets its own model off
the shared `openadmet_cyp_f2` FeatureSet (featurized through `smiles-to-2d-3d-v2` =
RDKit/Mordred 2D + curated xTB 3D).

The feature list comes from `Endpoint("smiles-to-2d-3d-v2").output_columns()`. That
returns descriptors only, so labels, `_ci_*`/`_std` columns, ids, `desc3d_*`
bookkeeping, and FeatureStore internals are excluded by construction — subtracting
label columns from `fs.columns` by hand gets this wrong in both directions.

The analog holdout is computed across all four targets with the same 50/10 parameters
the chemprop models use, so these models sit on the identical 529 held-out rows and
their `cyp_analog_holdout` captures are directly comparable.

Build the FeatureSet first: python cyp_feature_sets.py
"""

from workbench.api import Endpoint, FeatureSet, ModelFramework, ModelType
from workbench.training.splits import analog_holdout_split

FS_NAME = "openadmet_cyp_f2"
FEATURE_ENDPOINT = "smiles-to-2d-3d-v2"
VARIANT = "2d-3dv2"
BASE_TAGS = ["openadmet_cyp", "xgboost", "regression", "3dv2"]

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]

fs = FeatureSet(FS_NAME)
df = fs.pull_dataframe()
feats = [c for c in Endpoint(FEATURE_ENDPOINT).output_columns() if c in df.columns]

# Hold out potent hits and their close analogs across every target, mirroring the
# challenge's hit-expansion test set. Shared by all four models.
holdout_mask = analog_holdout_split(df, target_columns=TARGETS, n_hits=50, analogs_per_hit=10)
holdout = df[holdout_mask]
print(f"{len(feats)} features, analog holdout: {len(holdout)} of {len(df)} rows held out of training")

for iso in ISOFORMS:
    target = f"{iso}_pic50_direct_inhibition"
    name = f"cyp-{VARIANT}-reg-xgb-{iso.removeprefix('cyp')}"
    tags = BASE_TAGS + [VARIANT, iso]

    # Single-target frameworks cannot mask a NaN label, so rows without this isoform's
    # measurement leave the training view entirely. Coverage is sparse: most molecules
    # carry only one of the four isoforms.
    exclude_ids = list(df.loc[df[target].isna(), "molecule_name"])
    validation_ids = list(holdout.loc[holdout[target].notna(), "molecule_name"])
    print(f"{iso}: {len(df) - len(exclude_ids)} labeled rows, {len(validation_ids)} held out")

    model = fs.to_model(
        name=name,
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        feature_list=feats,
        target_column=target,
        description=f"CYP {iso.upper()} XGBoost UQ on {VARIANT} features, analog holdout",
        tags=tags,
        validation_ids=validation_ids,
        exclude_ids=exclude_ids,
    )
    model.set_owner("openadmet_cyp")

    end = model.to_endpoint(tags=tags)
    end.set_owner("openadmet_cyp")
    end.test_inference()
    end.cross_fold_inference()

    # Features are already in the FeatureSet, so the capture needs no re-inference. The
    # CI columns ride along so ST-RAE is scored against the challenge's own intervals.
    ci_cols = [f"{target}_ci_lower", f"{target}_ci_upper"]
    capture_df = holdout[["molecule_name", "smiles", target] + ci_cols + feats].dropna(subset=[target])
    end.inference(capture_df, capture_name="cyp_analog_holdout")
