"""What MultiTaskAlignment predicts about each auxiliary head, per scored isoform.

The union model carries ~19 auxiliary heads on one encoder and we have never had a prior
about which of them earn their place -- variants get built and read afterwards. MTA claims to
answer that from the data alone: per aux it scores the overlap region (do the targets agree
where both are measured) and the extension region (does the aux reach chemistry the primary
misses), and combines them into Use / Marginal / Risky / Skip.

This runs it once per scored isoform and tabulates the verdicts, so they are on record
*before* the variants are built. A prediction written down after the result is not a prediction.

Two things it cannot see, which is why its verdicts are read rather than obeyed:

  bounds        A censored label is a one-sided constraint, and MTA treats it as a
                measurement. Run this on the uncensored FeatureSet; on the censored one the
                ChEMBL heads would be scored on bound values as though they were potencies.
  resolution    Verdicts sit on hard thresholds (|Spearman| 0.4 and 0.95) applied to a point
                estimate. An aux near a boundary can flip on noise, and `n_shared` is
                reported here so a thin correlation is visible as thin.

    python cyp_aux_alignment.py                    # all four scored isoforms
    python cyp_aux_alignment.py --primary cyp2d6   # one
"""

import argparse

import pandas as pd
from workbench.algorithms.dataframe.multi_task_alignment import MultiTaskAlignment
from workbench.api import FeatureSet

FS_NAME = "openadmet_cyp_union_f1"
ISOFORMS = ["cyp1a2", "cyp2c9", "cyp2d6", "cyp3a4"]
PUBLIC_ISOFORMS = ISOFORMS + ["cyp2c19"]

SCORED = [f"{iso}_pic50_direct_inhibition" for iso in ISOFORMS]
AUX_FAMILIES = {
    "log2fc": [f"{iso}_log2fc" for iso in ISOFORMS],
    "tdi": [f"{iso}_pic50_tdi_condition" for iso in ISOFORMS],
    "emax": [f"{iso}_emax_vs_pos_ctrl_direct_inhibition" for iso in ISOFORMS],
    "chembl": [f"{iso}_pic50_chembl" for iso in PUBLIC_ISOFORMS],
    "veith": [f"{iso}_max_response" for iso in PUBLIC_ISOFORMS],
    "tox21": [f"{iso}_pic50_tox21" for iso in PUBLIC_ISOFORMS],
}


def family_of(aux: str) -> str:
    return next((name for name, cols in AUX_FAMILIES.items() if aux in cols), "other")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", choices=ISOFORMS, help="Only score this isoform (default: all four)")
    parser.add_argument("--out", default="outputs/aux_alignment.csv", help="Where to write the verdict table")
    args = parser.parse_args()

    df = FeatureSet(FS_NAME).pull_dataframe()
    every_aux = [c for cols in AUX_FAMILIES.values() for c in cols if c in df.columns]
    primaries = [f"{args.primary}_pic50_direct_inhibition"] if args.primary else SCORED

    rows = []
    for primary in primaries:
        # The other three scored targets are auxiliaries from this one's point of view.
        auxes = [c for c in every_aux + SCORED if c != primary]
        print(f"\n{'=' * 70}\n{primary}\n{'=' * 70}")
        mta = MultiTaskAlignment(
            df[["molecule_name", "smiles", primary] + auxes],
            primary=primary,
            auxiliaries=auxes,
            id_column="molecule_name",
        )
        summary = mta.summary()
        summary.insert(0, "primary", primary)
        summary["family"] = summary["aux"].map(family_of)
        rows.append(summary)
        cols = [
            "aux",
            "family",
            "n_shared",
            "n_aux_only",
            "spearman_r",
            "r_confidence",
            "tanimoto_coverage_mean",
            "overlap",
            "extension",
            "recommendation",
        ]
        print(summary[cols].to_string(index=False))

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(out)} aux/primary pairs)")
    print("\nVerdict counts by family:")
    print(pd.crosstab(out["family"], out["recommendation"]).to_string())


if __name__ == "__main__":
    main()
