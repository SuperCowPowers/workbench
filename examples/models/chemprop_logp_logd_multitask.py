"""LogP/LogD multi-task experiment: single-task LogD baseline vs multi-task LogD+LogP.

Pulls the curated overlap dataset from public data, builds a single FeatureSet,
then trains two chemprop models against it:

    1. Single-task LogD (baseline)        — target_column="logd"
    2. Multi-task LogD (primary) + LogP   — target_column=["logd", "logp"]

Both models train on the same FeatureSet. Single-task drops rows where logd is
NaN (the 1,558 LogP-only rows); multi-task keeps everything (chemprop's loss is
masked per-task). Same evaluation split for both -> apples-to-apples comparison.

Source data:
    s3://workbench-public-data/comp_chem/experiments/logp_logd_overlap.csv
    Built by data/public_data/build_logp_logd_overlap.py from the merged
    LogD (AstraZeneca) and LogP (OPERA + GraphormerLogP) public datasets.
"""

from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType, PublicData
from workbench.core.transforms.pandas_transforms.pandas_to_features import PandasToFeatures

# Recreate flag — flip to True to rebuild artifacts that already exist
recreate = False

FEATURESET_NAME = "logp_logd_overlap"
PUBLIC_DATA_KEY = "comp_chem/experiments/logp_logd_overlap"

# =============================================================================
# FeatureSet: pull curated overlap CSV from public data
# =============================================================================
if recreate or not FeatureSet(FEATURESET_NAME).exists():
    df = PublicData().get(PUBLIC_DATA_KEY)
    if df is None:
        raise RuntimeError(f"Could not load {PUBLIC_DATA_KEY} from public data")

    # The published CSV has only smiles/logp/logd; AWS Feature Store needs an id
    df = df.reset_index(drop=True)
    df["id"] = df.index.astype(int)

    print(f"Pushing {len(df):,} rows to FeatureSet '{FEATURESET_NAME}'")
    print(f"  has logp:  {df['logp'].notna().sum():,}")
    print(f"  has logd:  {df['logd'].notna().sum():,}")
    print(f"  has both:  {(df['logp'].notna() & df['logd'].notna()).sum():,}")

    to_features = PandasToFeatures(FEATURESET_NAME)
    to_features.set_input(df, id_column="id")
    to_features.set_output_tags(["chemprop", "logp", "logd", "multitask"])
    to_features.transform()

# =============================================================================
# Single-task: LogD only (baseline)
#   Chemprop drops rows where target is NaN, so the LogP-only rows are ignored
#   and this trains on the ~4,199 rows that have a logd value.
# =============================================================================
if recreate or not Model("logp-logd-chemprop-st").exists():
    fs = FeatureSet(FEATURESET_NAME)
    m = fs.to_model(
        name="logp-logd-chemprop-st",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="logd",  # Single target -> baseline
        feature_list=["smiles"],
        description="Single-task ChemProp LogD baseline (logp_logd_overlap)",
        tags=["chemprop", "logd", "single-task"],
    )
    m.set_owner("BW")

if recreate or not Endpoint("logp-logd-chemprop-st").exists():
    end = Model("logp-logd-chemprop-st").to_endpoint(tags=["chemprop", "logd", "single-task"])
    end.set_owner("BW")
    end.auto_inference()
    end.cross_fold_inference()

# =============================================================================
# Multi-task: LogD (primary) + LogP (auxiliary)
#   Same FeatureSet, all 5,757 rows. Chemprop's masked loss handles per-task NaN
#   automatically — no manual task weights needed (see chemprop.template).
#   LogD is listed first so it's the primary prediction in downstream UIs.
# =============================================================================
if recreate or not Model("logp-logd-chemprop-mt").exists():
    fs = FeatureSet(FEATURESET_NAME)
    m = fs.to_model(
        name="logp-logd-chemprop-mt",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column=["logd", "logp"],  # Multi-task: list, LogD primary
        feature_list=["smiles"],
        description="Multi-task ChemProp LogD (primary) + LogP (auxiliary)",
        tags=["chemprop", "logd", "logp", "multitask"],
    )
    m.set_owner("BW")

if recreate or not Endpoint("logp-logd-chemprop-mt").exists():
    end = Model("logp-logd-chemprop-mt").to_endpoint(tags=["chemprop", "logd", "logp", "multitask"])
    end.set_owner("BW")
    end.auto_inference()
    end.cross_fold_inference()
