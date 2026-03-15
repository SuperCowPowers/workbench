"""Partition AQSol data into overlap-based subsets for DatasetConcordance testing.

Uses a combined FingerprintProximity model (count fingerprints + Ruzicka distance)
to compute cross-dataset Tanimoto similarities — the same metric that DatasetConcordance
uses at runtime. Creates four subsets based on chemical space overlap:

    - base: 40% of compounds (the "reference" dataset)
    - high_overlap: 20% with highest max-Tanimoto to base
    - medium_overlap: 20% with moderate max-Tanimoto to base
    - low_overlap: 20% with lowest max-Tanimoto to base

The max-Tanimoto for each pool compound is the highest Tanimoto similarity to
ANY compound in the base set — this measures how well "covered" each compound
is by the reference dataset.

Outputs saved to S3: s3://workbench-public-data/comp_chem/aqsol_alignment/

Usage:
    python partition_aqsol_alignment.py
    python partition_aqsol_alignment.py --dry-run   # preview without uploading
"""

import argparse
import logging

import awswrangler as wr
import numpy as np
import pandas as pd

from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

S3_SOURCE = "s3://workbench-public-data/comp_chem/aqsol_public_data.csv"
S3_DEST = "s3://workbench-public-data/comp_chem/aqsol_alignment"


def main():
    parser = argparse.ArgumentParser(description="Partition AQSol data for alignment testing")
    parser.add_argument("--dry-run", action="store_true", help="Preview partition stats without uploading to S3")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load aqsol data
    log.info(f"Loading AQSol data from {S3_SOURCE}")
    df = wr.s3.read_csv(S3_SOURCE)
    df.columns = df.columns.str.lower()
    log.info(f"Loaded {len(df)} compounds")

    # Random split: 40% base, 60% candidate pool (split into 20% high/medium/low overlap)
    np.random.seed(args.seed)
    n = len(df)
    n_base = int(n * 0.4)
    indices = np.random.permutation(n)

    df_base = df.iloc[indices[:n_base]].reset_index(drop=True)
    df_pool = df.iloc[indices[n_base:]].reset_index(drop=True)

    log.info(f"Base set: {len(df_base)} compounds, Pool: {len(df_pool)} compounds")

    # Combine base + pool with a dataset column (same approach as DatasetConcordance)
    df_base_tagged = df_base.copy()
    df_pool_tagged = df_pool.copy()
    df_base_tagged["dataset"] = "base"
    df_pool_tagged["dataset"] = "pool"
    df_combined = pd.concat([df_base_tagged, df_pool_tagged], ignore_index=True)

    # Build ONE FingerprintProximity on combined data (count fingerprints + Ruzicka)
    # This ensures similarity values match what DatasetConcordance computes at runtime
    log.info("Building FingerprintProximity on combined data (count fingerprints)...")
    prox = FingerprintProximity(df_combined, id_column="id", include_all_columns=True)

    # Drop any pool compounds that were removed during fingerprint computation (bad SMILES)
    valid_ids = set(prox.df["id"])
    dropped = set(df_pool["id"]) - valid_ids
    if dropped:
        log.info(f"Dropped {len(dropped)} pool compounds with invalid SMILES: {dropped}")
        df_pool = df_pool[df_pool["id"].isin(valid_ids)].reset_index(drop=True)

    # For each pool compound, find its best base neighbor
    log.info("Computing cross-dataset 1-NN similarities (pool → base)...")
    pool_ids = df_pool["id"].tolist()
    neighbors_df = prox.neighbors(pool_ids, n_neighbors=20)

    # Filter to base-only neighbors
    dataset_lookup = prox.df.set_index("id")["dataset"]
    neighbors_df["neighbor_dataset"] = neighbors_df["neighbor_id"].map(dataset_lookup)
    base_neighbors = neighbors_df[neighbors_df["neighbor_dataset"] == "base"]

    # Extract max similarity per pool compound (1-NN similarity to base)
    max_sims = base_neighbors.groupby("id")["similarity"].max()
    pool_sims = max_sims.reindex(pool_ids, fill_value=0.0).values

    # Sort pool by similarity and split into thirds (~20% each of total)
    sorted_order = np.argsort(pool_sims)
    n_pool = len(sorted_order)
    third = n_pool // 3

    df_low = df_pool.iloc[sorted_order[:third]].reset_index(drop=True)
    df_medium = df_pool.iloc[sorted_order[third : 2 * third]].reset_index(drop=True)
    df_high = df_pool.iloc[sorted_order[2 * third :]].reset_index(drop=True)

    # Similarity arrays for summary stats (not stored on DataFrames)
    sims_low = pool_sims[sorted_order[:third]]
    sims_medium = pool_sims[sorted_order[third : 2 * third]]
    sims_high = pool_sims[sorted_order[2 * third :]]

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("PARTITION SUMMARY")
    log.info("=" * 70)
    log.info(f"  Base:           {len(df_base):>5} compounds")
    for name, sims in [("High overlap", sims_high), ("Medium overlap", sims_medium), ("Low overlap", sims_low)]:
        above_06 = (sims >= 0.6).sum()
        log.info(
            f"  {name:>15}: {len(sims):>5} compounds  "
            f"(tanimoto: {sims.min():.3f} – {sims.max():.3f}, "
            f"mean={sims.mean():.3f}, ≥0.6: {above_06})"
        )
    log.info("=" * 70)

    # Solubility distribution per partition (sanity check)
    log.info("\nSolubility distributions:")
    for name, subset in [("Base", df_base), ("High", df_high), ("Medium", df_medium), ("Low", df_low)]:
        sol = subset["solubility"]
        log.info(
            f"  {name:>8}: mean={sol.mean():.2f}, std={sol.std():.2f}, " f"range=[{sol.min():.2f}, {sol.max():.2f}]"
        )

    if args.dry_run:
        log.info("\n[DRY RUN] Skipping S3 upload")
        return

    # Save to S3
    log.info(f"\nUploading partitions to {S3_DEST}/")
    for name, subset in [
        ("aqsol_base", df_base),
        ("aqsol_high_overlap", df_high),
        ("aqsol_medium_overlap", df_medium),
        ("aqsol_low_overlap", df_low),
    ]:
        s3_path = f"{S3_DEST}/{name}.csv"
        wr.s3.to_csv(subset, s3_path, index=False)
        log.info(f"  Saved {name}: {len(subset)} rows → {s3_path}")

    log.info("\nDone!")


if __name__ == "__main__":
    main()
