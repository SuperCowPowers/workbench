"""Report LogP/LogD chemical-space overlap to inform multi-task modeling decisions.

Goal: decide whether a chemprop multi-task LogP+LogD model is likely to outperform
a LogD-only model. That depends on two things:

  (1) Coverage — do LogD compounds have nearby LogP neighbors in chemical space?
                 If LogP doesn't cover the LogD distribution, the auxiliary task adds
                 little signal where it matters.
  (2) Concordance — for similar compounds, do LogP and LogD agree (correlated)?
                 Strong agreement suggests easy auxiliary signal; mild disagreement
                 is exactly what multi-task learning can exploit.

Thin CLI wrapper around `workbench.algorithms.dataframe.MultiTaskAlignment`:

    MultiTaskAlignment(df, primary="logd", auxiliaries=["logp"], id_column="id")

We then print the per-aux summary and the multi-task verdict. Per-compound details
are written to `output/overlap_summary.csv` for downstream slicing/plotting.
"""

import logging
from pathlib import Path

import pandas as pd

from workbench.algorithms.dataframe import MultiTaskAlignment

log = logging.getLogger("workbench")

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "output" / "comp_chem"
LOGP_PATH = OUTPUT_DIR / "logp" / "logp_all.csv"
LOGD_PATH = OUTPUT_DIR / "logd" / "logd_all.csv"
SUMMARY_CSV = OUTPUT_DIR / "overlap_summary.csv"


def _print_summary(summary_row: pd.Series, n_logp: int, n_logd: int) -> None:
    print("\n" + "=" * 72)
    print("MultiTaskAlignment Summary  (primary=LogD, auxiliary=LogP)")
    print("=" * 72)
    print(f"  Primary (LogD):    {n_logd:>7,} compounds")
    print(f"  Auxiliary (LogP):  {n_logp:>7,} compounds")
    print(f"  Shared compounds:  {int(summary_row['n_shared']):>7,}")
    print(f"  LogP-only:         {int(summary_row['n_aux_only']):>7,}")

    print("\nLabel correlation on shared compounds")
    r = summary_row["pearson_r"]
    if pd.notna(r):
        print(f"  Pearson r = {r:.3f}  ({summary_row['r_confidence']} confidence)")
    else:
        print(f"  Pearson r = N/A  ({summary_row['r_confidence']})")

    print("\nChemical-space coverage (LogP -> nearest LogD)")
    print(f"  mean Tanimoto:        {summary_row['tanimoto_coverage_mean']:.3f}")
    print(f"  fraction sim >= 0.50: {100 * summary_row['frac_coverage_ge_05']:5.1f}%")
    print(f"  fraction sim >= 0.30: {100 * summary_row['frac_coverage_ge_03']:5.1f}%")

    print("\nZ-scored residual on aux-having rows")
    print(f"  |residual| mean: {summary_row['residual_abs_mean']:.3f}")
    print(f"  |residual| p95:  {summary_row['residual_abs_p95']:.3f}")


def _print_verdict(summary_row: pd.Series) -> None:
    print("\n" + "=" * 72)
    print("Multi-Task Recommendation")
    print("=" * 72)
    print(f"  Overlap verdict:    {summary_row['overlap']}")
    print(f"  Extension verdict:  {summary_row['extension']}")
    print(f"  Recommendation:     {summary_row['recommendation']}")


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

    # Build a wide multi-task DataFrame: one row per source compound with NaN where missing
    mt_df = pd.concat(
        [
            logd[["id", "smiles"]].assign(logd=logd["logd"]),
            logp[["id", "smiles"]].assign(logp=logp["logp"]),
        ],
        ignore_index=True,
    )

    mta = MultiTaskAlignment(
        mt_df,
        primary="logd",
        auxiliaries=["logp"],
        id_column="id",
    )

    summary = mta.summary()
    summary_row = summary.iloc[0]
    _print_summary(summary_row, n_logp=len(logp), n_logd=len(logd))
    _print_verdict(summary_row)

    results = mta.results()
    keep = [
        c
        for c in [
            "id",
            "smiles",
            "logd",
            "logp",
            "x",
            "y",
            "tanimoto_to_primary",
            "residual_logp",
        ]
        if c in results.columns
    ]
    results[keep].to_csv(SUMMARY_CSV, index=False)
    print(f"\nPer-compound results saved -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
