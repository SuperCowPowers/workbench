"""Producer: combined PXR FeatureSets (train + revealed phase-1 Analog Set 1).

Generic FeatureSets shared by every model. Each carries SMILES + 2D
(RDKit/Mordred) + 3D features; models select the subset they want via
`feature_list` (chemprop -> ["smiles"], xgb/pytorch -> the descriptor columns).
Two variants differ only in the 3D layer:

  - openadmet_pxr_f1  via smiles-to-2d-3d-v1  (v1 3D: 74 descriptors)
  - openadmet_pxr_f2  via smiles-to-2d-3d-v2  (v2 3D: curated, xTB-powered)

A `split` column marks each row ("train" or "phase1_test"):

  - phase-1 model holds the `phase1_test` rows out of training (via validation_ids)
    so the held-out set never trains it — then evaluates on exactly those rows.
  - phase-2 model trains on everything (train + phase-1) and predicts the blinded set.

f2 requires the smiles-to-2d-3d-v2 endpoint to be deployed first.

Run once before the phase models:  python pxr_feature_sets.py
"""

import pandas as pd
from workbench.api import DataSource, Endpoint, PublicData
from workbench.api.inference_cache import InferenceCache

# FeatureSet name -> the (Meta)Endpoint that produces its 2D+3D features.
FEATURE_SETS = {
    "openadmet_pxr_f1": "smiles-to-2d-3d-v1",  # v1 3D (74 descriptors)
    "openadmet_pxr_f2": "smiles-to-2d-3d-v2",  # v2 3D (curated, xTB)
}


def build_feature_set(fs_name: str, feature_endpoint: str, df: pd.DataFrame) -> None:
    """Featurize df through the (Meta)Endpoint and roll the result into a FeatureSet."""
    # SMILES-keyed cache (S3-persisted) so the expensive 3D (xTB) leg never recomputes.
    # Standardization tautomerizes `smiles`, so the original is preserved in `orig_smiles`
    # — key the cache on that to stay stable across runs.
    cached = InferenceCache(Endpoint(feature_endpoint), cache_key_column="smiles", output_key_column="orig_smiles")
    feat_df = cached.inference(df)  # append 2D + 3D feature columns
    DataSource(feat_df, name=f"{fs_name}_ds").to_features(
        fs_name, id_column="molecule_name", tags=["openadmet_pxr", "activity"]
    )
    n_train = (df.split == "train").sum()
    n_phase1 = (df.split == "phase1_test").sum()
    n_feat = len(cached.output_columns())
    print(
        f"Built '{fs_name}': {len(df)} rows ({n_train} train + {n_phase1} phase1_test), "
        f"{n_feat} 2D+3D features; cache holds {cached.cache_size():,} SMILES"
    )


train = PublicData().get("comp_chem/openadmet_pxr/pxr_train")[["molecule_name", "smiles", "pec50"]].copy()
train["split"] = "train"
phase1 = (
    PublicData().get("comp_chem/openadmet_pxr/pxr_test_phase1_unblinded")[["molecule_name", "smiles", "pec50"]].copy()
)
phase1["split"] = "phase1_test"
df = pd.concat([train, phase1]).dropna(subset=["pec50"]).drop_duplicates("molecule_name").reset_index(drop=True)

for name, ep in FEATURE_SETS.items():
    build_feature_set(name, ep, df)
