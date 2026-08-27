"""Producer: CYP FeatureSets for the OpenADMET blind challenge.

Generic FeatureSets shared by every model. Each carries SMILES + 2D
(RDKit/Mordred) + 3D features; models select the subset they want via
`feature_list` (chemprop -> ["smiles"], xgb/pytorch -> the descriptor columns).

Variants differ only in the 3D layer, and both scored tracks are built for each
variant so a model can be compared across 3D sets with nothing else moving:

  - openadmet_cyp_<v>        Regression track. 4,905 compounds, four sparse pIC50
                             targets. Each target carries `_ci_lower`/`_ci_upper`
                             (what `scripts/cyp_compare.py` scores ST-RAE against) and
                             `_std`.
  - openadmet_cyp_tdi_<v>    TDI track. 6,145 compounds, `cyp3a4_is_tdi` and
                             `cyp2d6_is_tdi` as binary targets.
Set `BUILD` to the variants this run should produce. A variant costs one xTB pass
over the 6,145 TDI compounds — the regression set is a strict subset of them, and
the InferenceCache is SMILES-keyed and S3-persisted, so both tracks and any rerun
draw from that one pass.

Target names keep the challenge's own naming, snake_cased. That is not cosmetic:
`scripts/cyp_compare.py` finds the credible interval by appending `_ci_lower` to the
target name, so renaming the targets silently drops ST-RAE.

The CI and std columns are in the FeatureSet so they can be scored against, and
they are NOT features. Every model script must pass an explicit `feature_list` --
an auto-generated one includes them, and a model handed the bounds that bracket
its own label scores near-perfectly and is worth nothing.

Run once before the model scripts:  python cyp_feature_sets.py
"""

import pandas as pd
from workbench.api import DataSource, Endpoint, PublicData
from workbench.api.inference_cache import InferenceCache
from workbench.utils.multi_task import validate_multi_task_data

# Variant suffix -> the (Meta)Endpoint that produces its 2D+3D features. v1 is the
# deprecated first-gen 3D set, kept as the comparison baseline; v2 is the curated xTB set.
FEATURE_VARIANTS = {
    "f1": "smiles-to-2d-3d-v1",  # v1 3D (74 descriptors)
    "f2": "smiles-to-2d-3d-v2",  # v2 3D (curated, xTB)
}

# Variants this run builds. Each is a multi-hour pass, so build them one at a time.
BUILD = ["f2"]

ISOFORMS = ["cyp3a4", "cyp2c9", "cyp2d6", "cyp1a2"]
TARGETS = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]


def build_feature_set(fs_name: str, feature_endpoint: str, df: pd.DataFrame, tags: list) -> None:
    """Featurize df through the (Meta)Endpoint and roll the result into a FeatureSet."""
    # SMILES-keyed cache (S3-persisted) so the expensive 3D (xTB) leg never recomputes.
    cached = InferenceCache(Endpoint(feature_endpoint))
    feat_df = cached.inference(df)  # append 2D + 3D feature columns
    DataSource(feat_df, name=f"{fs_name}_ds").to_features(fs_name, id_column="molecule_name", tags=tags)
    n_feat = len(cached.output_columns())
    print(f"Built '{fs_name}': {len(df)} rows, {n_feat} 2D+3D features; cache holds {cached.cache_size():,} SMILES")


# --- Regression track: four sparse pIC50 targets with credible intervals -------------

# Already one row per compound with NaN where an isoform has no fitted curve, which is
# the multi-task shape -- no combining needed.
inhibition = PublicData().get("comp_chem/openadmet/cyp/training/inhibition")
validate_multi_task_data(inhibition, TARGETS, id_column="molecule_name")

for variant in BUILD:
    build_feature_set(
        f"openadmet_cyp_{variant}",
        FEATURE_VARIANTS[variant],
        inhibition,
        tags=["openadmet_cyp", "multi_task", "activity"],
    )

coverage = ", ".join(f"{t.split('_')[0]}={inhibition[t].notna().sum()}" for t in TARGETS)
print(f"Regression targets: {coverage}")

# --- TDI track: binary labels for the two scored isoforms ----------------------------

# Labels only. `is_tdi` is derived from the shift between the direct and TDI arms, so
# carrying either arm's pIC50 alongside the label would hand the model the answer.
tdi = PublicData().get("comp_chem/openadmet/cyp/training/tdi")
tdi = tdi[["molecule_name", "smiles", "cyp3a4_is_tdi", "cyp2d6_is_tdi"]].copy()

for variant in BUILD:
    build_feature_set(
        f"openadmet_cyp_tdi_{variant}",
        FEATURE_VARIANTS[variant],
        tdi,
        tags=["openadmet_cyp", "tdi", "classification"],
    )

tdi_targets = ["cyp3a4_is_tdi", "cyp2d6_is_tdi"]
positives = ", ".join(f"{c.split('_')[0]}={int(tdi[c].sum())}/{int(tdi[c].notna().sum())}" for c in tdi_targets)
print(f"TDI positives: {positives}")
