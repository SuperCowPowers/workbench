"""Producer: one combined PXR FeatureSet (train + revealed phase-1 Analog Set 1).

A single, generic FeatureSet shared by both phase models. A `split` column marks
each row ("train" or "phase1_test"):

  - phase-1 model zero-weights the `phase1_test` rows (via sample_weights) so the
    held-out set never trains it — then evaluates on exactly those rows.
  - phase-2 model trains on everything (train + phase-1) and predicts the blinded set.

Run once before the phase models:  python pxr_feature_sets.py
"""

import pandas as pd
from workbench.api import DataSource, FeatureSet, PublicData

recreate = False
fs_name = "openadmet_pxr_f1"

if recreate or not FeatureSet(fs_name).exists():
    train = PublicData().get("comp_chem/openadmet_pxr/pxr_train")[["molecule_name", "smiles", "pec50"]].copy()
    train["split"] = "train"
    phase1 = (
        PublicData()
        .get("comp_chem/openadmet_pxr/pxr_test_phase1_unblinded")[["molecule_name", "smiles", "pec50"]]
        .copy()
    )
    phase1["split"] = "phase1_test"

    df = pd.concat([train, phase1]).dropna(subset=["pec50"]).drop_duplicates("molecule_name").reset_index(drop=True)
    DataSource(df, name="openadmet_pxr_combined_ds").to_features(
        fs_name, id_column="molecule_name", tags=["openadmet_pxr", "activity"]
    )
    n_train = (df.split == "train").sum()
    n_phase1 = (df.split == "phase1_test").sum()
    print(f"Built '{fs_name}': {len(df)} rows ({n_train} train + {n_phase1} phase1_test)")
else:
    print(f"FeatureSet '{fs_name}' exists — skipping (set recreate=True to rebuild)")
