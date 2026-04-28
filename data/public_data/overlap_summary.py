"""Report LogP/LogD chemical-space overlap to inform multi-task modeling decisions.

Goal: decide whether a chemprop multi-task LogP+LogD model is likely to outperform
a LogD-only model. That depends on two things:

  (1) Coverage — do LogD compounds have nearby LogP neighbors in chemical space?
                 If LogP doesn't cover the LogD distribution, the auxiliary task adds
                 little signal where it matters.
  (2) Concordance — for similar compounds, do LogP and LogD agree (correlated)?
                 Strong agreement suggests easy auxiliary signal; mild disagreement
                 is exactly what multi-task learning can exploit.

Thin CLI wrapper around `workbench.algorithms.dataframe.DatasetComparison`:

    DatasetComparison(logp, logd,
                      reference_target="logp", query_target="logd",
                      id_column="id")

We then print `dc.summary()` and the multi-task verdict. Per-compound details
are written to `output/overlap_summary.csv` for downstream slicing/plotting.
"""

import logging
from pathlib import Path

import pandas as pd

from workbench.algorithms.dataframe import DatasetComparison

log = logging.getLogger("workbench")

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "output"
LOGP_PATH = OUTPUT_DIR / "logp" / "logp_all.csv"
LOGD_PATH = OUTPUT_DIR / "logd" / "logd_all.csv"
SUMMARY_CSV = OUTPUT_DIR / "overlap_summary.csv"


def _print_summary(s: dict) -> None:
    print("\n" + "=" * 72)
    print("Dataset Comparison Summary  (reference=LogP, query=LogD)")
    print("=" * 72)
    print(f"  Reference (LogP):  {s['n_reference']:>7,} compounds")
    print(f"  Query (LogD):      {s['n_query']:>7,} compounds")

    t = s["tanimoto"]
    print(f"\nBest LogD->LogP Tanimoto (ECFP4)")
    print(f"  mean={t['mean']:.3f}  median={t['median']:.3f}  min={t['min']:.3f}  max={t['max']:.3f}")

    print("\nCoverage at similarity thresholds")
    for label, b in s["coverage"].items():
        print(f"  {label}  n={b['count']:>6,}  ({100 * b['fraction']:5.1f}% of LogD)")

    if s["residual"]:
        r = s["residual"]
        print("\nResidual = LogD - median(LogP near-neighbors)")
        print(f"  n={r['n']:,}  mean={r['mean']:+.3f}  median={r['median']:+.3f}  "
              f"|res|_mean={r['abs_mean']:.3f}  |res|_p95={r['abs_p95']:.3f}")

    if s["residual_by_sim_band"]:
        print("\nResidual by similarity band  (does agreement tighten as similarity rises?)")
        print(f"  {'sim band':>14s}  {'n':>6s}  {'mean res':>10s}  {'|res| mean':>11s}  {'|res| p95':>11s}")
        for b in reversed(s["residual_by_sim_band"]):  # descending sim
            band = f"[{b['sim_lo']:.2f}, {b['sim_hi']:.2f})"
            print(f"  {band:>14s}  {b['n']:>6,}  {b['residual_mean']:>+9.3f}  "
                  f"{b['residual_abs_mean']:>10.3f}  {b['residual_abs_p95']:>10.3f}")

    o = s["exact_smiles_overlap"]
    print(f"\nExact SMILES intersection: {o['count']:,}  "
          f"({100 * o['fraction_of_query']:.1f}% of LogD, {100 * o['fraction_of_reference']:.1f}% of LogP)")
    if "pearson" in o:
        print(f"  LogP vs LogD on shared compounds:")
        print(f"    Pearson correlation: {o['pearson']:.3f}")
        print(f"    Mean(LogP - LogD):   {o['mean_diff']:+.3f}")
        print(f"    |LogP - LogD| mean:  {o['abs_mean_diff']:.3f}")


def _multitask_verdict(s: dict) -> None:
    print("\n" + "=" * 72)
    print("Multi-Task Recommendation")
    print("=" * 72)
    cov_07 = s["coverage"].get(">= 0.70", {}).get("fraction", 0.0)
    cov_05 = s["coverage"].get(">= 0.50", {}).get("fraction", 0.0)
    corr = s["exact_smiles_overlap"].get("pearson", float("nan"))

    print(f"  LogD compounds with a LogP near-neighbor (Tanimoto >= 0.70): {100*cov_07:.1f}%")
    print(f"  LogD compounds with a LogP near-neighbor (Tanimoto >= 0.50): {100*cov_05:.1f}%")
    print(f"  LogP <-> LogD Pearson on exact-SMILES overlap: {corr:.3f}")
    print()
    if cov_05 > 0.8 and corr > 0.5:
        verdict = ("Strong overlap and decent LogP<->LogD correlation — a multi-task "
                   "chemprop model should benefit from the LogP auxiliary task.")
    elif cov_05 > 0.6:
        verdict = ("Moderate overlap — multi-task should help on the covered subset; "
                   "consider a hold-out split where LogD test compounds have LogP coverage.")
    else:
        verdict = ("Limited overlap — LogP and LogD occupy partially different chemical spaces. "
                   "Multi-task value will hinge on representation transfer rather than direct neighbor support.")
    print(f"  -> {verdict}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if not LOGP_PATH.exists():
        raise FileNotFoundError(f"{LOGP_PATH} not found — run pull_logp_data.py first.")
    if not LOGD_PATH.exists():
        raise FileNotFoundError(f"{LOGD_PATH} not found — run pull_logd_data.py first.")

    logp = pd.read_csv(LOGP_PATH)
    logd = pd.read_csv(LOGD_PATH)

    # ID disambiguation so the combined model can identify rows uniquely
    logp = logp.assign(id=logp["id"].astype(str).radd("logp_"))
    logd = logd.assign(id=logd["id"].astype(str).radd("logd_"))

    dc = DatasetComparison(
        logp[["id", "smiles", "logp"]],
        logd[["id", "smiles", "logd"]],
        reference_target="logp",
        query_target="logd",
        id_column="id",
    )

    s = dc.summary()
    _print_summary(s)
    _multitask_verdict(s)

    results = dc.results()
    keep = [c for c in ["id", "smiles", "dataset", "logp", "logd", "x", "y", "tanimoto_sim", "target_residual"]
            if c in results.columns]
    results[keep].to_csv(SUMMARY_CSV, index=False)
    print(f"\nPer-compound results saved -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
