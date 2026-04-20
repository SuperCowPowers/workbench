"""Create the PXR activity DataSource and three FeatureSets (2D / 3D / 2D+3D).

Pipeline:

    PublicData(pxr_train)
          │
          ▼
    DataSource: openadmet_pxr_activity     ← one source of truth (SMILES + pEC50)
          │
          ▼
    Run through feature endpoints:
      • smiles-to-2d-v1          direct call, fast
      • smiles-to-3d-full-v1     wrapped in InferenceCache — SMILES-keyed, persists
                                 across runs in S3 so we never recompute a compound's
                                 Boltzmann-ensemble 3D descriptors twice.
          │
     ┌────┴────┬─────────────┐
     ▼         ▼             ▼
   _2d       _3d          _2d_3d                     ← three FeatureSets
 (RDKit+    (Boltzmann    (concat of both
  Mordred    ensemble      descriptor sets)
  2D)        3D)

Usage (with WORKBENCH_CONFIG already set, e.g. ideaya_sandbox.json):

    python create_feature_sets.py                    # build anything missing
    python create_feature_sets.py --refeaturize      # clear the 3D InferenceCache and recompute
    python create_feature_sets.py --rebuild          # force-rebuild DataSource + all FeatureSets

Feature column lists come from each endpoint's FeatureEndpoint.feature_list()
(backed by ParameterStore, with auto-derive fallback on miss or staleness).
"""

import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd

from workbench.api import (
    DataSource,
    FeatureEndpoint,
    FeatureSet,
    PublicData,
)
from workbench.api.inference_cache import InferenceCache
from workbench.core.transforms.pandas_transforms import PandasToFeatures

log = logging.getLogger("openadmet_pxr.create_feature_sets")

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


# ─── Helpers ────────────────────────────────────────────────────────────────


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
    """Run df through the 2D then the 3D endpoint (3D is SMILES-cached).

    FeatureEndpoint handles both sides for us:
      - .feature_list() returns the registered feature columns from ParameterStore
      - .inference() routes to the right transport (realtime or async) based on
        how the endpoint was deployed — so we don't need to know or care.
    """
    fe_2d = FeatureEndpoint(ENDPOINT_2D)
    fe_3d = FeatureEndpoint(ENDPOINT_3D)
    cols_2d = fe_2d.feature_list()
    cols_3d = fe_3d.feature_list()
    log.info(f"  {ENDPOINT_2D}: {len(cols_2d)} registered features")
    log.info(f"  {ENDPOINT_3D}: {len(cols_3d)} registered features")

    # Compute our 2D Features (RDKit + Mordred)
    log.info(f"Running {len(df):,} rows through {ENDPOINT_2D} (2D RDKit+Mordred) …")
    df = fe_2d.inference(df)

    # Compute our 3D Features (Boltzmann ensemble)
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


def _ensure_feature_set(fs_name: str, df: pd.DataFrame, feature_cols: list[str], rebuild: bool) -> None:
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

    variant_tag = fs_name.rsplit("_", 1)[-1]  # "2d" / "3d" / "3d" for "...2d_3d"
    if fs_name == FEATURE_SET_2D_3D:
        variant_tag = "2d_3d"

    to_features = PandasToFeatures(fs_name)
    to_features.set_input(df_sub, id_column=ID_COL)
    to_features.set_output_tags(TAGS + [variant_tag])
    to_features.transform()
    FeatureSet(fs_name).set_owner("open_admet_pxr")


# ─── Entrypoint ─────────────────────────────────────────────────────────────


def main(rebuild: bool, refeaturize: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    raw_df = _pull_training_data()
    _ensure_data_source(raw_df, rebuild=rebuild)

    # Short-circuit if every FeatureSet is already present (and we're not rebuilding
    # or forcing refeaturization). Saves the 2D pass entirely.
    all_present = all(FeatureSet(n).exists() for n in (FEATURE_SET_2D, FEATURE_SET_3D, FEATURE_SET_2D_3D))
    if all_present and not rebuild and not refeaturize:
        log.info(
            "All three FeatureSets already exist — nothing to do. "
            "Use --rebuild to recreate, --refeaturize to clear the 3D cache."
        )
        return

    df, cols_2d, cols_3d = _featurize(raw_df, refeaturize=refeaturize)
    log.info(f"Feature column counts:  2D={len(cols_2d)}  3D={len(cols_3d)}  combined={len(cols_2d) + len(cols_3d)}")

    for fs_name, feature_cols in [
        (FEATURE_SET_2D, cols_2d),
        (FEATURE_SET_3D, cols_3d),
        (FEATURE_SET_2D_3D, cols_2d + cols_3d),
    ]:
        _ensure_feature_set(fs_name, df, feature_cols, rebuild=rebuild)

    log.info("Done. Created/verified:")
    log.info(f"  DataSource : {DATA_SOURCE_NAME}")
    for fs_name in (FEATURE_SET_2D, FEATURE_SET_3D, FEATURE_SET_2D_3D):
        log.info(f"  FeatureSet : {fs_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Force-rebuild DataSource and all FeatureSets.")
    parser.add_argument(
        "--refeaturize", action="store_true", help="Clear the 3D InferenceCache so all 3D features recompute."
    )
    args = parser.parse_args()
    main(rebuild=args.rebuild, refeaturize=args.refeaturize)
