"""Build tiered LogP/LogD multi-task overlap datasets for the chemprop experiment.

Produces one CSV per Tanimoto band of LogP-only auxiliary supervision:

    tanimoto in [0.7, 1.0)  -> output/experiments/logp_logd_overlap_07_10.csv  (high overlap)
    tanimoto in [0.3, 0.7)  -> output/experiments/logp_logd_overlap_03_07.csv  (medium)
    tanimoto in [0.0, 0.3)  -> output/experiments/logp_logd_overlap_00_03.csv  (low / "novel" auxiliary)

Each CSV is in chemprop multi-task format:

    smiles, logp, logd

with NaN where a value isn't measured.

Per-tier construction
---------------------
* Primary task (logd) is identical across all tiers: 4,199 LogD compounds.
* High tier (0.7-1.0, inclusive) naturally includes exact-SMILES compounds
  with Tanimoto = 1.0 — their LogP measurements and LogD measurements both
  appear in the merged CSV.
* Mid and low tier bands are below 1.0, so by definition their selections
  share no SMILES with LogD; rows with both targets populated do not appear
  in those tiers. That's the intent: each tier represents a distinct mode
  of auxiliary supervision (high = duplicate, mid = complementary, low = novel).
* Band selections capped at --max-aux (default 1,558) via deterministic
  random sample (--seed) when more candidates exist than the cap.

Approach: build a MultiTaskAlignment with logd as primary and logp as aux. Each
LogP-only row in the unified results gets a `tanimoto_to_primary` score. Filter
by band, cap, merge with the LogD primary, write CSV.

Output: output/experiments/logp_logd_overlap_<suffix>.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from workbench.algorithms.dataframe import MultiTaskAlignment

log = logging.getLogger("workbench")

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "output" / "comp_chem"
LOGP_PATH = OUTPUT_DIR / "logp" / "logp_all.csv"
LOGD_PATH = OUTPUT_DIR / "logd" / "logd_all.csv"
EXP_DIR = OUTPUT_DIR / "experiments"

DEFAULT_MAX_AUX = 4300  # near the natural ceiling of the high band ([0.7, 1.0] has ~4,318 LogP candidates)

TIERS = [
    ("07_10", 0.7, 1.0),
    ("03_07", 0.3, 0.7),
    ("00_03", 0.0, 0.3),
]


def build_one(
    mta: MultiTaskAlignment,
    logp: pd.DataFrame,
    logd: pd.DataFrame,
    suffix: str,
    tanimoto_min: float,
    tanimoto_max: float,
    max_aux: int,
    seed: int,
) -> Path:
    """Build a single tier CSV. Reuses the already-built MultiTaskAlignment model."""
    results = mta.results()
    # LogP-only rows in the unified results — `tanimoto_to_primary` is best Tanimoto to any LogD compound.
    logp_scored = results[results["logp"].notna()].copy()

    # Inclusive band [min, max]. The high tier ([0.7, 1.0]) naturally captures
    # exact-SMILES compounds (Tanimoto == 1.0); their merged rows show both
    # logp and logd populated. Lower tiers (< 1.0) by definition have no
    # smiles in common with LogD.
    band_mask = (logp_scored["tanimoto_to_primary"] >= tanimoto_min) & (
        logp_scored["tanimoto_to_primary"] <= tanimoto_max
    )
    band = logp_scored.loc[band_mask].copy()

    n_band_total = len(band)
    if n_band_total > max_aux:
        band = band.sample(n=max_aux, random_state=seed)
    log.info(
        f"[{suffix}]  Band [{tanimoto_min:.2f}, {tanimoto_max:.2f}]  "
        f"candidates={n_band_total:,}  selected={len(band):,}  "
        f"sim_range={band['tanimoto_to_primary'].min():.3f}-{band['tanimoto_to_primary'].max():.3f}"
    )

    # Take selected LogP rows and outer-merge with LogD primary.
    band_logp = logp[logp["smiles"].isin(band["smiles"])][["smiles", "logp"]]
    merged = band_logp.merge(logd[["smiles", "logd"]], on="smiles", how="outer")

    n_logp_only = (merged["logp"].notna() & merged["logd"].isna()).sum()
    n_logd_only = (merged["logp"].isna() & merged["logd"].notna()).sum()
    n_both = (merged["logp"].notna() & merged["logd"].notna()).sum()
    log.info(
        f"[{suffix}]  Merged: {len(merged):,} unique compounds  "
        f"(logp_only={n_logp_only:,}, logd_only={n_logd_only:,}, both={n_both:,})"
    )

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXP_DIR / f"logp_logd_overlap_{suffix}.csv"
    merged.to_csv(out_path, index=False)
    log.info(f"[{suffix}]  Saved -> {out_path}")
    return out_path


def main(max_aux: int, seed: int, only: list[str] | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if not LOGP_PATH.exists():
        raise FileNotFoundError(f"{LOGP_PATH} not found — run pull_logp_data.py first.")
    if not LOGD_PATH.exists():
        raise FileNotFoundError(f"{LOGD_PATH} not found — run pull_logd_data.py first.")

    logp = pd.read_csv(LOGP_PATH)
    logd = pd.read_csv(LOGD_PATH)
    log.info(f"Loaded LogP={len(logp):,}  LogD={len(logd):,}")

    # Disambiguate IDs across datasets so the combined fingerprint model can index uniquely.
    logp_ids = logp.assign(id=logp["id"].astype(str).radd("logp_"))
    logd_ids = logd.assign(id=logd["id"].astype(str).radd("logd_"))

    # Wide multi-task DataFrame: LogD-only rows + LogP-only rows. The shared FP/UMAP model
    # is built once, and every LogP row gets a tanimoto_to_primary score against LogD.
    mt_df = pd.concat(
        [
            logd_ids[["id", "smiles"]].assign(logd=logd_ids["logd"]),
            logp_ids[["id", "smiles"]].assign(logp=logp_ids["logp"]),
        ],
        ignore_index=True,
    )

    log.info("Building MultiTaskAlignment (primary=logd, aux=logp) ...")
    mta = MultiTaskAlignment(
        mt_df,
        primary="logd",
        auxiliaries=["logp"],
        id_column="id",
    )

    selected_tiers = TIERS if not only else [t for t in TIERS if t[0] in only]
    if not selected_tiers:
        raise ValueError(f"--only filtered out all tiers; valid suffixes: {[t[0] for t in TIERS]}")

    for suffix, lo, hi in selected_tiers:
        build_one(mta, logp, logd, suffix, lo, hi, max_aux, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max-aux",
        type=int,
        default=DEFAULT_MAX_AUX,
        help=f"Cap on LogP-only auxiliary rows per tier (default: {DEFAULT_MAX_AUX})",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling within a band when capped (default: 42)"
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=f"Build only specific tier(s) by suffix: {[t[0] for t in TIERS]}. Default builds all.",
    )
    args = parser.parse_args()
    main(args.max_aux, args.seed, args.only)
