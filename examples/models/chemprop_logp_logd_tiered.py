"""LogP/LogD multi-task experiment: tiered auxiliary by Tanimoto similarity.

Trains four chemprop models against tiered LogP/LogD datasets to test how the
auxiliary task's chemical-space overlap with the primary task affects multi-task
lift. Background: an earlier run on the high-overlap tier showed near-zero lift
(0.570 vs 0.583 RMSE), which is consistent with redundant supervision when LogP
and LogD measurements come from the same compounds (Pearson 0.967).

Hypothesis: MT lift on full_cross_fold should grow as the auxiliary moves out
of the LogD chemical neighborhood — representation transfer requires the
auxiliary to teach chemistry the primary doesn't already cover.

  1. Single-task LogD (baseline)
  2. Multi-task LogD + LogP, HIGH overlap   (Tanimoto 0.7 - 1.0)
  3. Multi-task LogD + LogP, MEDIUM overlap (Tanimoto 0.3 - 0.7)
  4. Multi-task LogD + LogP, LOW overlap    (Tanimoto 0.0 - 0.3)

Multi-task models down-weight the LogP auxiliary to 0.3 (LogD primary at 1.0)
to keep the primary task dominant in the gradient. Without this, auxiliary
gradient share exceeds 50% (4,300 LogP vs 4,199 LogD) and negative transfer
degrades LogD predictions across all tiers.

Source datasets (built by data/public_data/build_logp_logd_overlap.py):
    s3://workbench-public-data/comp_chem/experiments/logp_logd_overlap_07_10.csv
    s3://workbench-public-data/comp_chem/experiments/logp_logd_overlap_03_07.csv
    s3://workbench-public-data/comp_chem/experiments/logp_logd_overlap_00_03.csv
"""

from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType, PublicData
from workbench.core.transforms.pandas_transforms.pandas_to_features import PandasToFeatures

# Recreate flag — flip to True to rebuild artifacts that already exist
recreate = False

TIERS = [
    # (suffix, label,    description fragment)
    ("07_10", "high", "high overlap (Tanimoto 0.7-1.0)"),
    ("03_07", "med", "medium overlap (Tanimoto 0.3-0.7)"),
    ("00_03", "low", "low overlap (Tanimoto 0.0-0.3)"),
]


def ensure_featureset(suffix: str, description_fragment: str) -> str:
    """Ensure a tiered FeatureSet exists; pull from public data if missing."""
    fs_name = f"logp_logd_overlap_{suffix}"
    if not recreate and FeatureSet(fs_name).exists():
        return fs_name

    public_key = f"comp_chem/experiments/logp_logd_overlap_{suffix}"
    df = PublicData().get(public_key)
    if df is None:
        raise RuntimeError(f"Could not load {public_key} from public data")

    df = df.reset_index(drop=True)
    df["id"] = df.index.astype(int)

    print(f"Pushing {len(df):,} rows to FeatureSet '{fs_name}'  ({description_fragment})")
    print(f"  has logp:  {df['logp'].notna().sum():,}")
    print(f"  has logd:  {df['logd'].notna().sum():,}")
    print(f"  has both:  {(df['logp'].notna() & df['logd'].notna()).sum():,}")

    to_features = PandasToFeatures(fs_name)
    to_features.set_input(df, id_column="id")
    to_features.set_output_tags(["chemprop", "logp", "logd", "multitask", suffix])
    to_features.transform()
    return fs_name


# =============================================================================
# FeatureSets: pull each tier from public data and ingest
# =============================================================================
for suffix, _label, frag in TIERS:
    ensure_featureset(suffix, frag)

# =============================================================================
# Single-task LogD (baseline)
#   Uses the high-overlap FeatureSet but with target_column="logd" — chemprop
#   drops rows where target is NaN, so the LogP-only auxiliary rows are silently
#   ignored and this trains on the 4,199 LogD-bearing rows. The choice of which
#   tier's FeatureSet to use is arbitrary for the baseline (LogD primary rows
#   are identical across tiers); 07_10 is the most established and convenient.
# =============================================================================
if recreate or not Model("logp-logd-chemprop-st").exists():
    fs = FeatureSet("logp_logd_overlap_07_10")
    m = fs.to_model(
        name="logp-logd-chemprop-st",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="logd",  # Single target -> baseline; logp column ignored
        feature_list=["smiles"],
        description="Single-task ChemProp LogD baseline (4,199 LogD compounds)",
        tags=["chemprop", "logd", "single-task"],
    )
    m.set_owner("BW")

if recreate or not Endpoint("logp-logd-chemprop-st").exists():
    end = Model("logp-logd-chemprop-st").to_endpoint(tags=["chemprop", "logd", "single-task"])
    end.set_owner("BW")
    end.auto_inference()
    end.cross_fold_inference()

# =============================================================================
# Multi-task models — one per tier, LogD listed first (primary)
# =============================================================================
for suffix, label, frag in TIERS:
    model_name = f"logp-logd-chemprop-mt-{label}"
    fs_name = f"logp_logd_overlap_{suffix}"
    tags = ["chemprop", "logd", "logp", "multitask", label]

    if recreate or not Model(model_name).exists():
        fs = FeatureSet(fs_name)
        m = fs.to_model(
            name=model_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.CHEMPROP,
            target_column=["logd", "logp"],  # Multi-task: list, LogD primary
            feature_list=["smiles"],
            hyperparameters={"task_weights": [1.0, 0.3]},  # LogD=1.0, LogP=0.3
            description=f"Multi-task ChemProp LogD (primary) + LogP auxiliary, {frag}",
            tags=tags,
        )
        m.set_owner("BW")

    if recreate or not Endpoint(model_name).exists():
        end = Model(model_name).to_endpoint(tags=tags)
        end.set_owner("BW")
        end.auto_inference()
        end.cross_fold_inference()
