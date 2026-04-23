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
    fs = FeatureSet(fs_name)
    try:
        df = fs.query(f'SELECT {fs.id_column}, smiles, orig_smiles FROM {fs.table}')
    except Exception:
        continue
    nan_df = df[df["smiles"].isna()]
    if not len(nan_df):
        continue
    print(f"\n{'=' * 70}\nFeatureSet: {fs_name}\n{'=' * 70}")
    print(f"Total rows: {len(df)}, NaN smiles rows: {len(nan_df)}")
    print(nan_df.to_string())
