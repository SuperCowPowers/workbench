"""Benchmark the refactored FingerprintProximity on PublicData logp_all.

Validates the on-the-fly sparse Ruzicka path against a large reference set:
  - Build memory + time
  - Single-query latency (id-based)
  - Batch-query throughput (id-based)
  - Novel-SMILES query latency (was previously NotImplementedError for count FPs)
  - Per-row nn_distance / nn_id precomputed metrics
  - 2D projection cost (opt-out flag exercised)

Usage:
    WORKBENCH_CONFIG=/path/to/config.json python scripts/benchmark_fp_proximity.py
"""

import os
import time
import tracemalloc

import numpy as np
import pandas as pd

from workbench.api import PublicData
from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity


def timed(label):
    """Context-manager-ish timing wrapper. Returns elapsed seconds."""

    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.t0
            print(f"  [{label}] {self.elapsed:.3f}s")

    return _T()


def mem_peak_mb():
    """Return current peak tracemalloc usage in MB."""
    _, peak = tracemalloc.get_traced_memory()
    return peak / 1024 / 1024


def main():
    print("=" * 80)
    print("FingerprintProximity benchmark — PublicData logp_all")
    print("=" * 80)

    # Pull the dataset from PublicData
    print("\nPulling PublicData 'comp_chem/logp/logp_all'...")
    with timed("get_public_data"):
        pub_data = PublicData()
        df = pub_data.get("comp_chem/logp/logp_all")

    # Auto-detect id / target / fingerprint columns
    cols = list(df.columns)
    id_candidates = [c for c in cols if c.lower() in ("id", "udm_mol_bat_id", "molregno", "compound_id")]
    id_column = id_candidates[0] if id_candidates else cols[0]
    target_col = "logp" if "logp" in cols else next((c for c in cols if "logp" in c.lower()), None)
    if target_col is None:
        candidates = [c for c in cols if c not in (id_column, "fingerprint", "smiles")]
        target_col = candidates[0] if candidates else None

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
    print(f"  ID column: {id_column}")
    print(f"  Target column: {target_col}")

    if "fingerprint" in df.columns:
        sample_fp = str(df["fingerprint"].iloc[0])
        is_count = "," in sample_fp
        print(f"  Fingerprint format: {'count (Ruzicka)' if is_count else 'binary (Jaccard)'}")
        print(f"  Sample fingerprint (first 60 chars): {sample_fp[:60]}...")

    # ----- Build (proximity primitive only — what UQ inference will pay) -----
    print("\n[Build] FingerprintProximity (proximity primitive only)")
    tracemalloc.start()
    with timed("build_min") as t_build_min:
        prox = FingerprintProximity(
            df,
            id_column=id_column,
            target=target_col,
        )
    print(f"  Peak memory: {mem_peak_mb():.1f} MB")
    tracemalloc.stop()

    # ----- Build (lazy precompute via ActivityLandscape — landscape analysis cost) -----
    print("\n[Build] ActivityLandscape.proximity_stats() (triggers lazy nn_* precompute)")
    from workbench.algorithms.dataframe.activity_landscape import ActivityLandscape

    tracemalloc.start()
    with timed("build_precompute") as t_build_pre:
        landscape = ActivityLandscape(prox)
        _ = landscape.proximity_stats()  # forces lazy precompute
    print(f"  Peak memory: {mem_peak_mb():.1f} MB")
    tracemalloc.stop()

    # ----- Single-id query latency -----
    print("\n[Query] single-id latency (50 trials)")
    ids_pool = df[id_column].tolist()
    rng = np.random.default_rng(seed=42)
    sample_ids = rng.choice(ids_pool, size=50, replace=False)
    with timed("50× single-id query") as t_single:
        for sid in sample_ids:
            _ = prox.neighbors(sid, n_neighbors=10)
    print(f"  Per-query: {t_single.elapsed / 50 * 1000:.1f} ms")

    # ----- Batch-id query throughput -----
    for batch_size in (100, 1000, 5000):
        if batch_size > len(ids_pool):
            continue
        print(f"\n[Query] batch-id throughput (batch_size={batch_size})")
        batch_ids = rng.choice(ids_pool, size=batch_size, replace=False).tolist()
        with timed(f"batch {batch_size} ids") as t_batch:
            result = prox.neighbors(batch_ids, n_neighbors=10)
        print(f"  Per-query: {t_batch.elapsed / batch_size * 1000:.2f} ms")
        print(f"  Rows returned: {len(result):,}")

    # ----- Novel SMILES query (the regression we just fixed) -----
    if "smiles" in df.columns:
        novel_smiles = df["smiles"].iloc[: min(20, len(df))].tolist()
        print(f"\n[Query] novel-SMILES (n={len(novel_smiles)}) — previously blocked for count FPs")
        novel_query_df = pd.DataFrame({"smiles": novel_smiles, "query_id": novel_smiles})
        with timed(f"novel SMILES batch ({len(novel_smiles)})") as t_novel:
            novel_result = prox.neighbors_from_query_df(novel_query_df, n_neighbors=10)
        print(f"  Per-query: {t_novel.elapsed / len(novel_smiles) * 1000:.1f} ms")
        print(f"  Rows returned: {len(novel_result):,}")
        print("\n  First 3 results:")
        print(novel_result.head(3))
    else:
        print("\n[Query] novel-SMILES — skipped (no 'smiles' column in FeatureSet)")

    # ----- 2D projection cost (separate, since it materializes N×N) -----
    print("\n[project_2d] (materializes N×N transient distance matrix)")
    tracemalloc.start()
    with timed("project_2d") as t_proj:
        prox.project_2d()
    print(f"  Peak memory: {mem_peak_mb():.1f} MB")
    tracemalloc.stop()

    # ----- Summary -----
    n = len(df)
    print("\n" + "=" * 80)
    print(f"Summary for N={n:,} compounds:")
    print("=" * 80)
    print(f"  Build min-cost:           {t_build_min.elapsed:.2f}s")
    precompute_delta = t_build_pre.elapsed - t_build_min.elapsed
    print(f"  Build w/ precompute:      {t_build_pre.elapsed:.2f}s  (+{precompute_delta:.2f}s for nn metrics)")
    print(f"  Project 2D (lazy):        {t_proj.elapsed:.2f}s")
    print(f"  Single-id query (avg):    {t_single.elapsed / 50 * 1000:.1f} ms")
    if "smiles" in df.columns:
        print(f"  Novel SMILES (avg):       {t_novel.elapsed / len(novel_smiles) * 1000:.1f} ms")


if __name__ == "__main__":
    if not os.environ.get("WORKBENCH_CONFIG"):
        print("WARNING: WORKBENCH_CONFIG not set in environment")
    main()
