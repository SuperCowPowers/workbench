"""Pull the OpenADMET PXR Induction Challenge train/test data from HuggingFace.

Fetches every challenge CSV, snake_cases the column names (stripping units in
parentheses), and writes them to `output/openadmet_pxr/*.csv`. The shared
`upload_data.py` then publishes them to
`s3://workbench-public-data/comp_chem/openadmet_pxr/*.csv` and merges the
matching entries from `descriptions.json`.

Source (Apache-2.0, OpenADMET Consortium):
    https://huggingface.co/datasets/openadmet/pxr-challenge-train-test

Run:
    python pull_pxr_data.py            # fetch + write CSVs to output/openadmet_pxr/
    AWS_PROFILE=scp_sandbox_admin python upload_data.py --apply   # then publish

On each run this prints a per-file row count, column list, and a paste-ready
`columns` skeleton to help keep `descriptions.json` in sync with the real schema.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("workbench")

HF_REPO = "openadmet/pxr-challenge-train-test"
OUTPUT_DIR = Path(__file__).parent / "output" / "comp_chem" / "openadmet_pxr"

# Map: output CSV basename (== descriptions.json key, sans .csv) -> HF CSV filename
FILES = {
    "pxr_train": "pxr-challenge_TRAIN.csv",
    "pxr_test_blinded": "pxr-challenge_TEST_BLINDED.csv",
    "pxr_test_phase1_unblinded": "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    "pxr_counter_assay_train": "pxr-challenge_counter-assay_TRAIN.csv",
    "pxr_single_concentration_train": "pxr-challenge_single_concentration_TRAIN.csv",
    "pxr_96_compound_semi_pure_train": "pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv",
    "pxr_htchem_libraries_train": "pxr-challenge_htchem-libraries_TRAIN.csv",
    "pxr_structure_test_blinded": "pxr-challenge_structure_TEST_BLINDED.csv",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + snake_case column names (matches the rest of open_admet).

    Strips parenthesised units/annotations, collapses runs of underscores, and
    trims the ends, so a column like ``"pEC50 Std Error (-log10(molarity))"``
    becomes ``pec50_std_error``.
    """
    df = df.copy()
    cols = df.columns.str.lower()
    while cols.str.contains(r"\(").any():
        cols = cols.str.replace(r"\([^()]*\)", "", regex=True)
    cols = cols.str.replace(r"[^a-z0-9]+", "_", regex=True).str.replace(r"_+", "_", regex=True).str.strip("_")
    df.columns = cols
    return df


def fetch_from_hf(filename: str) -> pd.DataFrame:
    url = f"hf://datasets/{HF_REPO}/{filename}"
    log.info(f"Fetching {url}")
    return normalize_columns(pd.read_csv(url))


def main():
    parser = argparse.ArgumentParser(description="Pull the OpenADMET PXR challenge datasets from HuggingFace")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--only", nargs="*", choices=list(FILES.keys()), help="Only pull these basenames (default: all)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    basenames = args.only or list(FILES.keys())
    skeleton = {}
    print("\n" + "=" * 70)
    print("OpenADMET PXR Data Pull")
    print("=" * 70)
    for basename in basenames:
        df = fetch_from_hf(FILES[basename])
        out_path = args.output_dir / f"{basename}.csv"
        df.to_csv(out_path, index=False)
        log.info(f"  {basename}.csv  ({len(df):,} rows, {len(df.columns)} cols)  cols={df.columns.tolist()}")
        skeleton[f"{basename}.csv"] = {"num_compounds": int(len(df)), "columns": {c: "" for c in df.columns}}

    print(f"\nWrote {len(basenames)} files -> {args.output_dir}")
    print("\ndescriptions.json skeleton (fill in column meanings, merge into descriptions.json):")
    print(json.dumps(skeleton, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
