"""Build the PXR activity DataSource and the three descriptor FeatureSets.

Producer stage of the openadmet_pxr pipeline DAG. Pulls the public PXR training
data, runs it through the 2D and 3D feature endpoints, and rolls the results up
into three FeatureSets that the model scripts consume:

    PublicData(pxr_train)
          │
          ▼
    DataSource: openadmet_pxr_activity        (SMILES + pEC50 + meta)
          │
    feature endpoints:
      • smiles-to-2d-v1        RDKit + Mordred 2D (direct call)
      • smiles-to-3d-full-v1   Boltzmann-ensemble 3D, wrapped in InferenceCache
                               (SMILES-keyed, persists in S3 — never recompute)
          │
     ┌────┴────┬──────────────┐
     ▼         ▼              ▼
    _2d       _3d           _2d_3d            (three FeatureSets)

Outputs: fs:openadmet_pxr_activity_2d, fs:openadmet_pxr_activity_3d,
         fs:openadmet_pxr_activity_2d_3d

Run:
    python pxr_feature_sets.py                # build anything missing
    python pxr_feature_sets.py --refeaturize  # clear the 3D InferenceCache and recompute
    python pxr_feature_sets.py --rebuild      # force-rebuild DataSource + all FeatureSets
"""

import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd

from workbench.api import DataSource, Endpoint, FeatureSet, PublicData
from workbench.api.inference_cache import InferenceCache
from workbench.core.transforms.pandas_transforms import PandasToFeatures

log = logging.getLogger("workbench")

# ─── Names ──────────────────────────────────────────────────────────────────
DATA_SOURCE_NAME = "openadmet_pxr_activity"
FEATURE_SET_2D = "openadmet_pxr_activity_2d"
FEATURE_SET_3D = "openadmet_pxr_activity_3d"
FEATURE_SET_2D_3D = "openadmet_pxr_activity_2d_3d"

# Columns
ID_COL = "molecule_name"
SMILES_COL = "smiles"
TARGET_COL = "pec50"
META_COLS = ["ocnt_id", "pec50_std_error"]  # passthrough for uncertainty-weighted training later

# Feature endpoints
ENDPOINT_2D = "smiles-to-2d-v1"  # RDKit + Mordred 2D (salts removed) — fast, no cache needed
ENDPOINT_3D = "smiles-to-3d-full-v1"  # Boltzmann-weighted 3D ensemble (74 features) — cached

TAGS = ["openadmet_pxr", "activity"]


def _pull_training_data() -> pd.DataFrame:
    """Pull the PXR training set from the public bucket, keeping only what we need."""
    log.info("Pulling pxr_train from workbench-public-data …")
    df = PublicData().get("comp_chem/openadmet_pxr/pxr_train")
    keep = [ID_COL, SMILES_COL, TARGET_COL] + [c for c in META_COLS if c in df.columns]
    df = df[keep].dropna(subset=[TARGET_COL]).reset_index(drop=True)
    log.info(f"  {len(df):,} rows, columns={list(df.columns)}")
    return df


def _ensure_data_source(df: pd.DataFrame, rebuild: bool) -> None:
    """Create (or refresh) the single PXR activity DataSource."""
    if DataSource(DATA_SOURCE_NAME).exists() and not rebuild:
        log.info(f"DataSource '{DATA_SOURCE_NAME}' already exists — reusing")
        return

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / f"{DATA_SOURCE_NAME}.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Creating DataSource '{DATA_SOURCE_NAME}' from {len(df):,} rows …")
        ds = DataSource(str(csv_path), name=DATA_SOURCE_NAME)
        ds.set_owner("open_admet_pxr")


def _featurize(df: pd.DataFrame, refeaturize: bool) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Run df through the 2D then the 3D endpoint (3D is SMILES-cached)."""
    fe_2d = Endpoint(ENDPOINT_2D)
    fe_3d = Endpoint(ENDPOINT_3D)
    cols_2d = fe_2d.output_columns()
    cols_3d = fe_3d.output_columns()
    log.info(f"  {ENDPOINT_2D}: {len(cols_2d)} registered features")
    log.info(f"  {ENDPOINT_3D}: {len(cols_3d)} registered features")

    log.info(f"Running {len(df):,} rows through {ENDPOINT_2D} (2D RDKit+Mordred) …")
    df = fe_2d.inference(df)

    cached_3d = InferenceCache(fe_3d, cache_key_column=SMILES_COL)
    if refeaturize:
        log.info(f"--refeaturize: clearing InferenceCache for {ENDPOINT_3D}")
        cached_3d.clear_cache()

    log.info(
        f"Running {len(df):,} rows through {ENDPOINT_3D} (3D Boltzmann, cached). "
        f"Current cache size: {cached_3d.cache_size():,} SMILES."
    )
    df = cached_3d.inference(df)
    log.info(f"  {ENDPOINT_3D}: cache now holds {cached_3d.cache_size():,} SMILES")

    return df, cols_2d, cols_3d


def _ensure_feature_set(fs_name: str, df: pd.DataFrame, feature_cols: list[str], variant_tag: str, rebuild: bool) -> None:
    """Create a FeatureSet with the given feature columns, if missing or rebuilding."""
    if FeatureSet(fs_name).exists() and not rebuild:
        log.info(f"FeatureSet '{fs_name}' already exists — skipping")
        return

    passthrough = [ID_COL, SMILES_COL, TARGET_COL] + [c for c in META_COLS if c in df.columns]
    keep = passthrough + feature_cols
    df_sub = df[keep].dropna(subset=[TARGET_COL]).copy()
    log.info(
        f"Creating FeatureSet '{fs_name}'  —  "
        f"{len(df_sub):,} rows × {len(feature_cols)} features (+ {len(passthrough)} passthrough)"
    )
    to_features = PandasToFeatures(fs_name)
    to_features.set_input(df_sub, id_column=ID_COL)
    to_features.set_output_tags(TAGS + [variant_tag])
    to_features.transform()
    FeatureSet(fs_name).set_owner("open_admet_pxr")


def main(rebuild: bool, refeaturize: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    raw_df = _pull_training_data()
    _ensure_data_source(raw_df, rebuild=rebuild)

    all_present = all(FeatureSet(n).exists() for n in (FEATURE_SET_2D, FEATURE_SET_3D, FEATURE_SET_2D_3D))
    if all_present and not rebuild and not refeaturize:
        log.info("All three FeatureSets already exist — nothing to do (use --rebuild / --refeaturize).")
        return

    df, cols_2d, cols_3d = _featurize(raw_df, refeaturize=refeaturize)
    log.info(f"Feature column counts:  2D={len(cols_2d)}  3D={len(cols_3d)}  combined={len(cols_2d) + len(cols_3d)}")

    for fs_name, feature_cols, tag in [
        (FEATURE_SET_2D, cols_2d, "2d"),
        (FEATURE_SET_3D, cols_3d, "3d"),
        (FEATURE_SET_2D_3D, cols_2d + cols_3d, "2d_3d"),
    ]:
        _ensure_feature_set(fs_name, df, feature_cols, tag, rebuild=rebuild)

    log.info("Done. DataSource + FeatureSets ready:")
    for fs_name in (FEATURE_SET_2D, FEATURE_SET_3D, FEATURE_SET_2D_3D):
        log.info(f"  FeatureSet : {fs_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Force-rebuild DataSource and all FeatureSets.")
    parser.add_argument("--refeaturize", action="store_true", help="Clear the 3D InferenceCache so all 3D features recompute.")
    args = parser.parse_args()
    main(rebuild=args.rebuild, refeaturize=args.refeaturize)
