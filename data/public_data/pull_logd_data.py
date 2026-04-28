"""Pull and merge publicly available experimental LogD datasets into a unified CSV.

Sources (octanol-water distribution coefficient, experimental measurements):
  1. AstraZeneca / ChEMBL  (~4.2K compounds, logD @ pH 7.4)

The AstraZeneca data is fetched directly from the MoleculeNet S3 mirror
(no PyTDC dependency — that package drags in scikit-learn just to download
the same static CSV). Same data is also redistributed by Therapeutic Data
Commons as `Lipophilicity_AstraZeneca`.

Standardization (RDKit + ChEMBL pipeline) matches pull_logp_data.py exactly,
so the resulting `smiles` column can be inner-joined to the LogP dataset to
overlap compounds reported across both endpoints.

Output:
  output/logd/logd_all.csv              -- merged, deduplicated on canonical SMILES
  output/logd/logd_<source>.csv         -- per-source files
"""

import argparse
import io
import logging
from pathlib import Path

import pandas as pd

from alignment_utils import run_alignment_checks
from pull_common import download, merge_and_deduplicate, standardize_df

log = logging.getLogger("workbench")

OUTPUT_DIR = Path(__file__).parent / "output" / "logd"
VALUE_NAME = "logd"
FILE_PREFIX = "logd"


def pull_astrazeneca_chembl() -> pd.DataFrame:
    """AstraZeneca / ChEMBL logD @ pH 7.4 (~4.2K compounds).

    AstraZeneca-measured logD values from ChEMBL, mirrored on the MoleculeNet
    S3 bucket (also redistributed via DeepChem and Therapeutic Data Commons).
    """
    log.info("Pulling AstraZeneca / ChEMBL logD @ pH 7.4 ...")

    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
    raw = download(url, desc="Lipophilicity.csv")
    df = pd.read_csv(io.BytesIO(raw))
    log.info(f"  Raw: {len(df)} rows, columns={df.columns.tolist()}")

    return standardize_df(
        df,
        smiles_col="smiles",
        value_col="exp",
        source="astrazeneca_chembl",
        value_name=VALUE_NAME,
    )


SOURCES = {
    "astrazeneca": pull_astrazeneca_chembl,
}


def main():
    parser = argparse.ArgumentParser(description="Pull and merge public LogD datasets")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=list(SOURCES.keys()),
        choices=list(SOURCES.keys()),
        help="Which sources to pull (default: all)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name in args.sources:
        try:
            df = SOURCES[name]()
            log.info(f"  {name}: {len(df)} rows")
            frames.append(df)
        except Exception:
            log.exception(f"Failed to pull {name}")

    if not frames or all(len(f) == 0 for f in frames):
        log.error("No data pulled from any source!")
        return

    merged = merge_and_deduplicate(frames, output_dir=args.output_dir, value_name=VALUE_NAME, file_prefix=FILE_PREFIX)
    out_path = args.output_dir / f"{FILE_PREFIX}_all.csv"
    merged.to_csv(out_path, index=False)
    log.info(f"Saved merged dataset -> {out_path}  ({len(merged)} unique compounds)")

    print("\n" + "=" * 60)
    print("LogD Data Pull Summary")
    print("=" * 60)
    for source, grp in pd.concat(frames, ignore_index=True).groupby("source"):
        print(f"  {source:<35s} {len(grp):>7,} compounds")
    print(f"  {'TOTAL (deduplicated)':<35s} {len(merged):>7,} compounds")
    print(f"\nOutput: {args.output_dir}")
    print("=" * 60)

    # Report overlap with LogP if it has been pulled
    logp_path = Path(__file__).parent / "output" / "logp" / "logp_all.csv"
    if logp_path.exists():
        logp_smiles = set(pd.read_csv(logp_path, usecols=["smiles"])["smiles"])
        overlap = merged["smiles"].isin(logp_smiles).sum()
        print(f"\nOverlap with logp_all.csv: {overlap:,} / {len(merged):,} compounds ({100*overlap/len(merged):.1f}%)")
        print("=" * 60)

    run_alignment_checks(
        df=merged,
        value_col=VALUE_NAME,
        std_col=f"{VALUE_NAME}_std",
        count_col=f"{VALUE_NAME}_count",
        sources_col="sources",
        assay_name="LogD",
        expected_range=(-3, 7),
        expected_mean=(1, 4),
    )


if __name__ == "__main__":
    main()
