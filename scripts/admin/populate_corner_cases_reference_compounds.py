"""Populate the public reference-compounds dataset for pipeline corner-case tests.

Builds a curated table of edge-case SMILES inputs that stress the pipeline's
input handling — empty strings, invalid parses, overly long SMILES, isotopes,
and unusual atoms — with the expected desc3d_status value for each.

Dataset location:
    s3://workbench-public-data/comp_chem/reference_compounds/corner_cases.csv

Access (read-only, unsigned):
    from workbench.api import PublicData
    df = PublicData().get("comp_chem/reference_compounds/corner_cases")

Consumed by:
    tests/feature_endpoints/test_3d_corner_cases.py

Usage:
    python scripts/admin/populate_corner_cases_reference_compounds.py --dry-run
    python scripts/admin/populate_corner_cases_reference_compounds.py
"""

import argparse
import logging

import pandas as pd

from reference_compounds_common import run_populate

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

CSV_KEY = "comp_chem/reference_compounds/corner_cases.csv"

# Note on long-SMILES case:
#   standardize() rejects SMILES > 1000 chars before passing to RDKit, so the
#   "too_long" case tests the upstream guard. The 3D pipeline never sees the
#   molecule — expected_status reflects that upstream rejection producing an
#   empty 'smiles' column after standardization, which compute_descriptors_3d
#   treats as skip:empty.
REFERENCE_COMPOUNDS = [
    # --- Input-handling corner cases ---
    {
        "name": "empty_string",
        "smiles": "",
        "expected_status": "skip:empty",
        "notes": "Empty SMILES — handled gracefully by skip:empty branch",
    },
    {
        "name": "invalid_parse",
        "smiles": "THIS_IS_NOT_VALID_SMILES",
        "expected_status": "skip:empty",
        "notes": "Unparseable SMILES — standardize rejects (None); 3D endpoint sees empty",
    },
    {
        "name": "too_long",
        "smiles": "C" * 1200,
        "expected_status": "skip:empty",
        "notes": ">1000 char guard — standardize rejects before RDKit sees it",
    },
    # --- Isotopes / unusual atoms that should still compute normally ---
    {
        "name": "deuterated_methanol",
        "smiles": "[2H]C([2H])([2H])O",
        "expected_status": "ok",
        "notes": "Deuterium labels preserved; 3D pipeline handles isotopes transparently",
    },
    {
        "name": "phenylboronic_acid",
        "smiles": "OB(O)c1ccccc1",
        "expected_status": "ok",
        "notes": "Boron is a supported atom type — phenylboronic acid parses and embeds",
    },
    {
        "name": "thioether",
        "smiles": "CSC",
        "expected_status": "ok",
        "notes": "Sulfur heteroatom control — simple thioether computes normally",
    },
]

COLUMN_ORDER = ["id", "name", "smiles", "expected_status", "notes"]


def build_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(REFERENCE_COMPOUNDS)
    df.insert(0, "id", range(len(df)))
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMN_ORDER]


DESCRIPTION = {
    "description": (
        "Reference compounds for pipeline corner-case tests. Each row pairs "
        "an edge-case SMILES input with the expected desc3d_status value "
        "produced by the full endpoint pipeline (standardize + "
        "compute_descriptors_3d). Covers empty input, invalid parses, "
        "overly-long strings, isotopes, and unusual atoms."
    ),
    "columns": {
        "id": "Integer row index",
        "name": "Short identifier for the corner case",
        "smiles": "Edge-case SMILES input fed to the pipeline",
        "expected_status": (
            "Expected value of desc3d_status after the full endpoint pipeline "
            "(standardize + compute_descriptors_3d). 'ok' means the molecule "
            "computes successfully; 'skip:empty' means it was rejected before "
            "descriptor computation."
        ),
        "notes": "Why this corner case is in the reference set",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Populate the corner-cases reference-compounds dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print the DataFrame without uploading")
    args = parser.parse_args()

    df = build_dataframe()
    DESCRIPTION["num_compounds"] = int(len(df))
    run_populate(df, CSV_KEY, DESCRIPTION, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
