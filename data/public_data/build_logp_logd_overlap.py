"""Build the LogP/LogD multi-task overlap dataset for the chemprop experiment.

Takes all LogD compounds plus the top-K LogP compounds (by Tanimoto similarity to
the LogD set) and writes them to a single CSV in chemprop multi-task format:

    smiles, logp, logd

One row per unique canonical SMILES. NaN where a value isn't measured. Compounds
present in both datasets get both columns populated — those carry the strongest
multi-task signal.

Approach: run DatasetComparison the *opposite* direction from overlap_summary.py
(logd = reference, logp = query). The query Tanimoto then scores each LogP by
its similarity to the LogD set, which is exactly the picking criterion.

Output: output/experiments/logp_logd_overlap.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from workbench.algorithms.dataframe import DatasetComparison

log = logging.getLogger("workbench")

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "output"
LOGP_PATH = OUTPUT_DIR / "logp" / "logp_all.csv"
LOGD_PATH = OUTPUT_DIR / "logd" / "logd_all.csv"
EXP_DIR = OUTPUT_DIR / "experiments"
OUT_CSV = EXP_DIR / "logp_logd_overlap.csv"

DEFAULT_TOP_K = 5000


def main(top_k: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if not LOGP_PATH.exists():
        raise FileNotFoundError(f"{LOGP_PATH} not found — run pull_logp_data.py first.")
    if not LOGD_PATH.exists():
        raise FileNotFoundError(f"{LOGD_PATH} not found — run pull_logd_data.py first.")

    logp = pd.read_csv(LOGP_PATH)
    logd = pd.read_csv(LOGD_PATH)
    log.info(f"Loaded LogP={len(logp):,}  LogD={len(logd):,}")

    # Disambiguate IDs across datasets so the combined fingerprint model can index uniquely
    logp_ids = logp.assign(id=logp["id"].astype(str).radd("logp_"))
    logd_ids = logd.assign(id=logd["id"].astype(str).radd("logd_"))

    log.info("Building DatasetComparison (logd=reference, logp=query) ...")
    dc = DatasetComparison(
        df_reference=logd_ids[["id", "smiles", "logd"]],
        df_query=logp_ids[["id", "smiles", "logp"]],
        reference_target="logd",
        query_target="logp",
        id_column="id",
    )

    results = dc.results()
    logp_scored = results[results["dataset"] == "query"].copy()
    logp_scored = logp_scored.sort_values("tanimoto_sim", ascending=False).head(top_k)
    log.info(f"Top-{top_k} LogP picked (sim range {logp_scored['tanimoto_sim'].min():.3f}"
             f" – {logp_scored['tanimoto_sim'].max():.3f})")

    # Outer-merge on canonical SMILES — one row per unique compound, NaN where missing
    logp_subset = logp[logp["smiles"].isin(logp_scored["smiles"])][["smiles", "logp"]]
    merged = logp_subset.merge(logd[["smiles", "logd"]], on="smiles", how="outer")

    n_logp_only = merged["logp"].notna().sum() - (merged["logp"].notna() & merged["logd"].notna()).sum()
    n_logd_only = merged["logd"].notna().sum() - (merged["logp"].notna() & merged["logd"].notna()).sum()
    n_both = (merged["logp"].notna() & merged["logd"].notna()).sum()

    log.info(f"Merged dataset: {len(merged):,} unique compounds")
    log.info(f"  - LogP only:  {n_logp_only:,}")
    log.info(f"  - LogD only:  {n_logd_only:,}")
    log.info(f"  - Both:       {n_both:,}")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)
    log.info(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of LogP compounds to retain by similarity (default: {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()
    main(args.top_k)
