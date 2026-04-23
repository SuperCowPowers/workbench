"""Scan Workbench FeatureSets for NaN SMILES.

Usage:
    WORKBENCH_CONFIG=/Users/briford/.workbench/ideaya_prod.json \
    AWS_PROFILE=idb-prod-admin \
    python smiles_nan_investigation.py
"""
from workbench.api import FeatureSet
from workbench.cached.cached_meta import CachedMeta

fs_names = CachedMeta().feature_sets()["Feature Group"].tolist()

for fs_name in fs_names:
    print(f"\n{'=' * 70}\nFeatureSet: {fs_name}\n{'=' * 70}")
    fs = FeatureSet(fs_name)
    try:
        df = fs.query(f'SELECT {fs.id_column}, smiles, orig_smiles FROM {fs.table}')
    except Exception as e:
        print(f"  Skipped: {e}")
        continue
    nan_df = df[df["smiles"].isna()]
    print(f"Total rows: {len(df)}, NaN smiles rows: {len(nan_df)}")

    # Print out each row with a NaN SMILES for investigation
    if len(nan_df):
        print(nan_df.to_string())
