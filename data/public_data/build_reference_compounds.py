"""Build the static reference-compound datasets used by the comp-chem tests.

The curated compounds live as JSON (one file per dataset) under
``reference_compounds/``; this script turns each into a CSV under
``output/<csv_key>``. Keeping the data out of Python keeps the long SMILES strings
out of the linter's way and makes each dataset diff-friendly and editable on its own.
Descriptions live in ``descriptions.json`` like every other dataset.

Each JSON holds the full dataset:
    {
      "csv_key":   "comp_chem/reference_compounds/<name>.csv",
      "columns":   [...],            # output column order (includes "id")
      "compounds": [ {...}, ... ]    # one dict per row (no "id"; added here)
    }

All compounds are PUBLIC: named drugs, CAS-referenced chemicals, or obviously
synthetic test strings.

Usage:
    python build_reference_compounds.py                       # all
    python build_reference_compounds.py --dataset 3d_perf     # one (repeatable)
    AWS_PROFILE=scp_sandbox_admin python upload_data.py --apply
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = Path(__file__).parent
SOURCE_DIR = DATA_DIR / "reference_compounds"
OUTPUT_DIR = DATA_DIR / "output"


def available_datasets() -> dict:
    """Map dataset name -> JSON path, discovered from the source directory."""
    return {p.stem: p for p in sorted(SOURCE_DIR.glob("*.json"))}


def build_dataframe(compounds: list, columns: list) -> pd.DataFrame:
    """Compounds list -> DataFrame: prepend an integer id, pad any declared-but-
    absent columns with NA, and order to the dataset's column schema."""
    df = pd.DataFrame(compounds)
    df.insert(0, "id", range(len(df)))
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


def build_one(json_path: Path) -> None:
    cfg = json.loads(json_path.read_text(encoding="utf-8"))
    df = build_dataframe(cfg["compounds"], cfg["columns"])
    out_path = OUTPUT_DIR / cfg["csv_key"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"{json_path.stem}: {len(df)} rows -> {out_path.relative_to(DATA_DIR)}")


def main():
    datasets = available_datasets()
    parser = argparse.ArgumentParser(description="Build reference-compound CSVs")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(datasets),
        help="Dataset to build (repeatable). Default: all.",
    )
    args = parser.parse_args()

    for name in args.dataset or sorted(datasets):
        build_one(datasets[name])


if __name__ == "__main__":
    main()
