"""Populate ALL static reference-compound datasets used by the comp-chem tests.

The curated compound data lives as JSON (one file per dataset) under
``scripts/admin/reference_compounds/``; this script is just the loader/uploader.
Keeping the data out of Python keeps the long SMILES strings out of the linter's
way and makes each dataset diff-friendly and editable on its own.

Each JSON holds the full dataset:
    {
      "csv_key":     "comp_chem/reference_compounds/<name>.csv",
      "columns":     [...],            # output column order (includes "id")
      "description": {...},            # patched into the public descriptions.json
      "compounds":   [ {...}, ... ]    # one dict per row (no "id"; added here)
    }

All compounds are PUBLIC: named drugs, CAS-referenced chemicals, or obviously
synthetic test strings.

Activity-cliffs data is NOT here: it pulls live SMILES + measured values from a
Model/FeatureSet and emits one CSV per cliff, so it stays in its own script
(populate_activity_cliffs_reference_compounds.py).

Usage:
    python scripts/admin/populate_reference_compounds.py --dry-run          # all, no upload
    python scripts/admin/populate_reference_compounds.py                    # all -> S3
    python scripts/admin/populate_reference_compounds.py --dataset 3d_perf  # one (repeatable)
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from reference_compounds_common import run_populate

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = Path(__file__).parent / "reference_compounds"


def available_datasets() -> dict:
    """Map dataset name -> JSON path, discovered from the data directory."""
    return {p.stem: p for p in sorted(DATA_DIR.glob("*.json"))}


def build_dataframe(compounds: list, columns: list) -> pd.DataFrame:
    """Compounds list -> DataFrame: prepend an integer id, pad any declared-but-
    absent columns with NA, and order to the dataset's column schema."""
    df = pd.DataFrame(compounds)
    df.insert(0, "id", range(len(df)))
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


def populate_one(json_path: Path, dry_run: bool) -> None:
    cfg = json.loads(json_path.read_text(encoding="utf-8"))
    df = build_dataframe(cfg["compounds"], cfg["columns"])
    description = dict(cfg["description"])
    description["num_compounds"] = int(len(df))
    log.info(f"\n=== {json_path.stem} ===")
    run_populate(df, cfg["csv_key"], description, dry_run=dry_run)


def main():
    datasets = available_datasets()
    parser = argparse.ArgumentParser(description="Populate reference-compound datasets")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(datasets),
        help="Dataset to populate (repeatable). Default: all.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print without uploading")
    args = parser.parse_args()

    for name in args.dataset or sorted(datasets):
        populate_one(datasets[name], args.dry_run)


if __name__ == "__main__":
    main()
