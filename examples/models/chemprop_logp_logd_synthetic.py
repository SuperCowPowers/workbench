"""LogP/LogD synthetic multi-task experiment: chemprop MT implementation validation.

Real-data MT experiments showed no LogD lift on top of single-task baselines, plausibly
because the real LogP/LogD overlap has Pearson 0.967 (auxiliary nearly redundant). This
script validates the chemprop MT wiring independent of data quality, using deterministic
synthetic LogP labels on a 1k-row subset.

Trains five models against synthetic 1k datasets:

  1. ST baseline                  — single-task LogD on 1k real LogD compounds
  2. MT crippen   (Recipe A)      — auxiliary = Crippen.MolLogP, Pearson(aux, LogD) ~0.39
  3. MT blended   (Recipe B)      — auxiliary = Crippen + aromaticity + rotatable, ~0.42
  4. MT strong    (Recipe C)      — auxiliary = RandomForest predicted LogD, ~0.97 (training)
  5. MT real      (Recipe D)      — auxiliary = REAL LogD on 3,199 extended LogD compounds

Recipe D is the cross-the-board guaranteed-lift recipe: the auxiliary head is supervised
by real LogD on chemistry the primary never sees. MT-real has access to ~4x more
LogD-relevant supervision than ST. If MT-real doesn't beat ST on every metric, the
chemprop multi-task wiring is broken — there is no other defensible explanation.

Each MT model uses task_weights=[1.0, 0.3] to keep LogD primary in the gradient. LogD is
listed first in target_column for downstream UI consistency.

Source datasets (built by data/public_data/build_synthetic_multi_task.py):
    comp_chem/synthetic/multi_task/log_d            (1,000 real LogD compounds)
    comp_chem/synthetic/multi_task/log_p            (Recipe A: pure Crippen)
    comp_chem/synthetic/multi_task/log_p_blended    (Recipe B: blended chemistry)
    comp_chem/synthetic/multi_task/log_p_strong     (Recipe C: RF-teacher LogD)
    comp_chem/synthetic/multi_task/log_p_real       (Recipe D: real LogD on extended)
"""

import pandas as pd

from workbench.api import Endpoint, FeatureSet, Model, ModelFramework, ModelType, PublicData
from workbench.core.transforms.pandas_transforms.pandas_to_features import PandasToFeatures

# Recreate flag — flip to True to rebuild artifacts that already exist
recreate = False

PUBLIC_PREFIX = "comp_chem/synthetic/multi_task"

RECIPES = [
    # (label,    logp_public_key,    description fragment)
    ("crippen", "log_p", "pure Crippen.MolLogP"),
    ("blended", "log_p_blended", "Crippen + aromaticity + rotatable bonds"),
    ("strong", "log_p_strong", "RandomForest-predicted LogD (synthetic teacher)"),
    ("real", "log_p_real", "real LogD on extended chemistry (guaranteed-lift control)"),
]


def _build_merged_mt_featureset(label: str, logp_key: str, frag: str) -> str:
    """Pull log_d + log_p_<recipe>, concatenate into multi-task layout, push to FeatureSet."""
    fs_name = f"logp_logd_synth_{label}"
    if not recreate and FeatureSet(fs_name).exists():
        return fs_name

    logd_df = PublicData().get(f"{PUBLIC_PREFIX}/log_d")
    logp_df = PublicData().get(f"{PUBLIC_PREFIX}/{logp_key}")
    if logd_df is None or logp_df is None:
        raise RuntimeError(f"Could not load synthetic data from public data ({logp_key})")

    # Concat: each row has either logd or logp populated; chemistries are disjoint
    # (LogD set is from LogD corpus, LogP set is mid-tier compounds not in LogD).
    merged = pd.concat([logd_df, logp_df], ignore_index=True)
    merged = merged.reset_index(drop=True)
    merged["id"] = merged.index.astype(int)
    merged = merged[["id", "smiles", "logd", "logp"]]

    print(f"Pushing {len(merged):,} rows to FeatureSet '{fs_name}'  ({frag})")
    print(f"  has logd:  {merged['logd'].notna().sum():,}")
    print(f"  has logp:  {merged['logp'].notna().sum():,}")
    print(f"  has both:  {(merged['logd'].notna() & merged['logp'].notna()).sum():,}")

    to_features = PandasToFeatures(fs_name)
    to_features.set_input(merged, id_column="id")
    to_features.set_output_tags(["chemprop", "logd", "logp", "synthetic", "multitask", label])
    to_features.transform()
    return fs_name


def _build_st_featureset() -> str:
    """LogD-only FeatureSet for the single-task baseline (1k compounds)."""
    fs_name = "logp_logd_synth_st"
    if not recreate and FeatureSet(fs_name).exists():
        return fs_name

    logd_df = PublicData().get(f"{PUBLIC_PREFIX}/log_d")
    if logd_df is None:
        raise RuntimeError(f"Could not load {PUBLIC_PREFIX}/log_d from public data")

    logd_df = logd_df.reset_index(drop=True)
    logd_df["id"] = logd_df.index.astype(int)

    print(f"Pushing {len(logd_df):,} rows to FeatureSet '{fs_name}'  (ST baseline)")
    to_features = PandasToFeatures(fs_name)
    to_features.set_input(logd_df, id_column="id")
    to_features.set_output_tags(["chemprop", "logd", "synthetic", "single-task"])
    to_features.transform()
    return fs_name


# =============================================================================
# FeatureSets: ST baseline + one per MT recipe
# =============================================================================
_build_st_featureset()
for label, logp_key, frag in RECIPES:
    _build_merged_mt_featureset(label, logp_key, frag)

# =============================================================================
# Single-task LogD baseline
# =============================================================================
if recreate or not Model("logp-logd-synth-st").exists():
    fs = FeatureSet("logp_logd_synth_st")
    m = fs.to_model(
        name="logp-logd-synth-st",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="logd",
        feature_list=["smiles"],
        description="Single-task ChemProp LogD baseline (1k synthetic dataset)",
        tags=["chemprop", "logd", "synthetic", "single-task"],
    )
    m.set_owner("BW")

if recreate or not Endpoint("logp-logd-synth-st").exists():
    end = Model("logp-logd-synth-st").to_endpoint(tags=["chemprop", "logd", "synthetic", "single-task"])
    end.set_owner("BW")
    end.auto_inference()
    end.cross_fold_inference()

# =============================================================================
# Multi-task models — one per recipe, LogD listed first (primary)
# =============================================================================
for label, _logp_key, frag in RECIPES:
    model_name = f"logp-logd-synth-mt-{label}"
    fs_name = f"logp_logd_synth_{label}"
    tags = ["chemprop", "logd", "logp", "synthetic", "multitask", label]

    if recreate or not Model(model_name).exists():
        fs = FeatureSet(fs_name)
        m = fs.to_model(
            name=model_name,
            model_type=ModelType.UQ_REGRESSOR,
            model_framework=ModelFramework.CHEMPROP,
            target_column=["logd", "logp"],  # Multi-task: LogD primary
            feature_list=["smiles"],
            hyperparameters={"task_weights": [1.0, 0.3]},  # LogD=1.0, LogP=0.3
            description=f"Synthetic-aux MT ChemProp, {frag}",
            tags=tags,
        )
        m.set_owner("BW")

    if recreate or not Endpoint(model_name).exists():
        end = Model(model_name).to_endpoint(tags=tags)
        end.set_owner("BW")
        end.auto_inference()
        end.cross_fold_inference()
