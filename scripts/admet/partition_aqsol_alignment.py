"""Partition AQSol data into overlap-based subsets for DatasetConcordance testing.

Uses a combined FingerprintProximity model (count fingerprints + Ruzicka distance)
to compute cross-dataset Tanimoto similarities — the same metric that DatasetConcordance
uses at runtime. Creates four subsets based on chemical space overlap:

    - base: 40% of compounds (the "reference" dataset)
    - high_overlap: all pool compounds (sim 0–1.0, realistic mix)
    - medium_overlap: pool compounds with sim <= 0.6 (novel + moderate)
    - low_overlap: pool compounds with sim <= 0.35 (truly novel only)

Each overlap level is a cumulative range — higher levels are supersets of lower ones.
This mirrors real-world scenarios where even high-overlap datasets contain some novel compounds.

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

from workbench.api import PublicData
from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# FIXME: Switch S3_DEST to use PublicData() for writes
S3_DEST = "s3://workbench-public-data/comp_chem/aqsol_alignment"

# Cumulative similarity thresholds for overlap partitions
LOW_THRESHOLD = 0.35
MEDIUM_THRESHOLD = 0.6


def main():
    parser = argparse.ArgumentParser(description="Partition AQSol data for alignment testing")
    parser.add_argument("--dry-run", action="store_true", help="Preview partition stats without uploading to S3")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load aqsol data
    log.info("Loading AQSol data via PublicData")
    df = PublicData().get("comp_chem/aqsol/aqsol_public_data")
    df.columns = df.columns.str.lower()
    log.info(f"Loaded {len(df)} compounds")

    # Random split: 40% base, 60% candidate pool
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

    # Iteratively smooth solubility using fingerprint-based neighbors.
    # This creates a clean SAR landscape for test data where structurally similar compounds
    # have similar targets, so synthetic alignment perturbation dominates.
    # Multiple passes aggressively flatten the natural SAR noise.
    all_ids = prox.df["id"].tolist()
    original_std = prox.df["solubility"].std()
    n_passes = 5
    k_smooth = 50
    log.info(f"Smoothing solubility: {n_passes} passes, k={k_smooth} fingerprint neighbors...")

    for i in range(n_passes):
        nbrs = prox.neighbors(all_ids, n_neighbors=k_smooth)
        # Weighted mean: use similarity as weight (higher sim = more influence)
        nbrs["weighted_target"] = nbrs["solubility"] * nbrs["similarity"]
        smoothed = nbrs.groupby("id").agg(
            weighted_sum=("weighted_target", "sum"),
            weight_total=("similarity", "sum"),
        )
        smoothed["solubility_smooth"] = smoothed["weighted_sum"] / smoothed["weight_total"]

        # Apply smoothed values back to prox.df (used by next iteration's neighbors call)
        smooth_lookup = smoothed["solubility_smooth"]
        prox.df["solubility"] = prox.df["id"].map(smooth_lookup)
        log.info(f"  Pass {i + 1}: std={prox.df['solubility'].std():.3f}")

    # Apply final smoothed values to the partition DataFrames
    df_base["solubility"] = df_base["id"].map(smooth_lookup)
    df_pool["solubility"] = df_pool["id"].map(smooth_lookup)
    log.info(f"Smoothed solubility: std={prox.df['solubility'].std():.3f} (was {original_std:.3f})")

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

    # Cumulative threshold partitions (each higher level is a superset)
    low_mask = pool_sims <= LOW_THRESHOLD
    medium_mask = pool_sims <= MEDIUM_THRESHOLD
    # high = all pool compounds (no filter)

    df_low = df_pool[low_mask].reset_index(drop=True)
    df_medium = df_pool[medium_mask].reset_index(drop=True)
    df_high = df_pool.reset_index(drop=True)  # full pool

    sims_low = pool_sims[low_mask]
    sims_medium = pool_sims[medium_mask]
    sims_high = pool_sims

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("PARTITION SUMMARY (cumulative thresholds)")
    log.info("=" * 70)
    log.info(f"  Base:           {len(df_base):>5} compounds")
    for name, sims, thresh in [
        ("Low overlap", sims_low, f"sim <= {LOW_THRESHOLD}"),
        ("Medium overlap", sims_medium, f"sim <= {MEDIUM_THRESHOLD}"),
        ("High overlap", sims_high, "sim <= 1.0 (all)"),
    ]:
        if len(sims) > 0:
            log.info(
                f"  {name:>15}: {len(sims):>5} compounds  "
                f"({thresh}, range: {sims.min():.3f}–{sims.max():.3f}, mean={sims.mean():.3f})"
            )
        else:
            log.info(f"  {name:>15}:     0 compounds  ({thresh})")
    log.info("=" * 70)

    # Solubility distribution per partition (sanity check)
    log.info("\nSolubility distributions:")
    for name, subset in [("Base", df_base), ("High", df_high), ("Medium", df_medium), ("Low", df_low)]:
        if len(subset) > 0:
            sol = subset["solubility"]
            log.info(
                f"  {name:>8}: mean={sol.mean():.2f}, std={sol.std():.2f}, "
                f"range=[{sol.min():.2f}, {sol.max():.2f}], n={len(subset)}"
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
