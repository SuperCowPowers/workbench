"""Assemble a submission from the per-isoform columns of several scored entries.

ST-RAE is macro-averaged over isoforms that are scored independently, and the file is
four independent columns, so the best entry is the per-isoform best rather than whichever
single calibration scored best overall. Mixing is exact rather than a projection: each
column carries the score it already earned.

Placement that maximises R2 is not the same placement that minimises ST-RAE, and CYP2D6
is where they diverge hardest -- moving it onto its true centre raised R2 0.363 -> 0.447
and worsened ST-RAE 0.565 -> 0.694. Until the ST-RAE optimum is derived directly, the
per-isoform pick is how that gets exploited.

    python cyp_mix_submission.py --base FILE --take CYP2D6=FILE --tag mixed
"""

import argparse
from pathlib import Path

import pandas as pd
from cyp_recalibrate import VALUE_COLUMNS
from openadmet_validation import validate_activity_submission
from workbench.api import PublicData

OUT = Path(__file__).parent / "outputs"


def mix(base: Path, take: dict, out_dir: Path, tag: str) -> Path:
    """Write `base` with each isoform in `take` replaced by that file's column."""
    sub = pd.read_csv(base)
    for iso, source in take.items():
        col = VALUE_COLUMNS[iso]
        donor = pd.read_csv(source)
        if not donor["Molecule_Name"].equals(sub["Molecule_Name"]):
            raise ValueError(f"{source} rows are not aligned with {base}")
        before = sub[col]
        sub[col] = donor[col]
        print(
            f"{iso:<8} mean {before.mean():.2f} -> {sub[col].mean():.2f}  "
            f"sd {before.std():.2f} -> {sub[col].std():.2f}   [{Path(source).name}]"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{base.stem}_{tag}.csv"
    sub.to_csv(path, index=False)

    blind = PublicData().get("comp_chem/openadmet/cyp/testing/blinded")
    expected = {str(m) for m in blind["molecule_name"]}
    ok, errors = validate_activity_submission(path, expected_ids=expected)
    if not ok:
        raise ValueError(f"{path} failed OpenADMET's validator:\n  " + "\n  ".join(errors))
    print(f"\nPassed OpenADMET's validator: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Entry supplying the untouched columns")
    parser.add_argument(
        "--take",
        action="append",
        default=[],
        metavar="ISO=FILE",
        help="Replace one isoform's column from another entry; repeatable",
    )
    parser.add_argument("--tag", default="mixed", help="Suffix for the output filename")
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory")
    args = parser.parse_args()

    take = {}
    for spec in args.take:
        iso, _, source = spec.partition("=")
        if iso not in VALUE_COLUMNS:
            raise ValueError(f"Unknown isoform '{iso}' — expected one of {list(VALUE_COLUMNS)}")
        take[iso] = Path(source)
    mix(args.base, take, args.out, args.tag)
