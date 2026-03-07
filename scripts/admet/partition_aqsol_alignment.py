"""Partition AQSol data into overlap-based subsets for DatasetAlignment testing.

Uses Workbench's FingerprintProximity to compute Tanimoto similarities and creates
four subsets based on chemical space overlap:

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

    # Build FingerprintProximity on base set
    log.info("Building FingerprintProximity on base set...")
    prox = FingerprintProximity(df_base, id_column="id")

    # Query pool compounds against the base to get 1-NN similarities
    log.info("Computing nearest-neighbor similarities (pool → base)...")
    pool_smiles = df_pool["smiles"].tolist()
    neighbors_df = prox.neighbors_from_smiles(pool_smiles, n_neighbors=1)

    # Extract max similarity per pool compound (1-NN similarity)
    max_sims = neighbors_df.groupby("query_id")["similarity"].max().reindex(pool_smiles, fill_value=0.0).values

    # Sort pool by similarity and split into thirds (~20% each of total)
    sorted_order = np.argsort(max_sims)
    n_pool = len(sorted_order)
    third = n_pool // 3

    df_low = df_pool.iloc[sorted_order[:third]].reset_index(drop=True)
    df_medium = df_pool.iloc[sorted_order[third : 2 * third]].reset_index(drop=True)
    df_high = df_pool.iloc[sorted_order[2 * third :]].reset_index(drop=True)

    # Add the max-Tanimoto column to query sets (useful for validation)
    df_low["max_tanimoto_to_base"] = max_sims[sorted_order[:third]]
    df_medium["max_tanimoto_to_base"] = max_sims[sorted_order[third : 2 * third]]
    df_high["max_tanimoto_to_base"] = max_sims[sorted_order[2 * third :]]

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("PARTITION SUMMARY")
    log.info("=" * 70)
    log.info(f"  Base:           {len(df_base):>5} compounds")
    for name, subset in [("High overlap", df_high), ("Medium overlap", df_medium), ("Low overlap", df_low)]:
        sims = subset["max_tanimoto_to_base"]
        log.info(
            f"  {name:>15}: {len(subset):>5} compounds  "
            f"(max_tanimoto: {sims.min():.3f} – {sims.max():.3f}, "
            f"mean={sims.mean():.3f})"
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
