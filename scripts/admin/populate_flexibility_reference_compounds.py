"""Populate the public reference-compounds dataset used by flexibility-comparison tests.

Builds a table of compounds spanning the rotatable-bond spectrum (0 → 19) so
tests can compare fast-mode (10 conformers) vs Boltzmann-mode (50-300 adaptive
conformers) output and surface the real divergence for flexible molecules.

Dataset location:
    s3://workbench-public-data/comp_chem/reference_compounds/flexibility.csv

Access (read-only, unsigned):
    from workbench.api import PublicData
    df = PublicData().get("comp_chem/reference_compounds/flexibility")

Consumed by:
    tests/feature_endpoints/test_3d_flexibility_comparison.py

Usage:
    python scripts/admin/populate_flexibility_reference_compounds.py --dry-run
    python scripts/admin/populate_flexibility_reference_compounds.py
"""

import argparse
import logging

import pandas as pd

from reference_compounds_common import run_populate

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

CSV_KEY = "comp_chem/reference_compounds/flexibility.csv"

# Compounds selected to span the rotatable-bond spectrum. Rigid compounds let
# us prove the two endpoints implement the same descriptor math (they should
# agree tightly). Flexible compounds let us see how much Boltzmann's denser
# conformer sampling actually shifts ensemble-averaged descriptors.
REFERENCE_COMPOUNDS = [
    # --- Rigid (0 rotatable bonds): endpoints should agree very tightly ---
    {
        "name": "benzene",
        "smiles": "c1ccccc1",
        "rot_bonds": 0,
        "flexibility_class": "rigid",
        "notes": "Single aromatic ring, 0 rot bonds — all conformers identical",
    },
    {
        "name": "naphthalene",
        "smiles": "c1ccc2ccccc2c1",
        "rot_bonds": 0,
        "flexibility_class": "rigid",
        "notes": "Fused bicyclic aromatic, 0 rot bonds",
    },
    {
        "name": "adamantane",
        "smiles": "C1C2CC3CC1CC(C2)C3",
        "rot_bonds": 0,
        "flexibility_class": "rigid",
        "notes": "T_d-symmetric cage, 0 rot bonds",
    },
    {
        "name": "caffeine",
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "rot_bonds": 0,
        "flexibility_class": "rigid",
        "notes": "Rigid planar drug, 0 rot bonds",
    },
    # --- Semi-rigid (2-4 rot bonds): mild divergence expected ---
    {
        "name": "aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "rot_bonds": 2,
        "flexibility_class": "semi_rigid",
        "notes": "Aromatic core with 2 rot bonds in the acetate + COOH",
    },
    {
        "name": "hexane",
        "smiles": "CCCCCC",
        "rot_bonds": 3,
        "flexibility_class": "semi_rigid",
        "notes": "Small alkane, 3 rot bonds — modest conformational spread",
    },
    {
        "name": "ibuprofen_s",
        "smiles": "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1",
        "rot_bonds": 4,
        "flexibility_class": "semi_rigid",
        "notes": "Drug with isobutyl + carboxylic-acid chains, 4 rot bonds",
    },
    # --- Flexible (7-9 rot bonds): measurable divergence ---
    {
        "name": "decane",
        "smiles": "CCCCCCCCCC",
        "rot_bonds": 7,
        "flexibility_class": "flexible",
        "notes": "C10 alkane, 7 rot bonds — extended vs compact conformers differ",
    },
    {
        "name": "dodecane",
        "smiles": "CCCCCCCCCCCC",
        "rot_bonds": 9,
        "flexibility_class": "flexible",
        "notes": "C12 alkane, 9 rot bonds — Boltzmann tier 200 conformers",
    },
    # --- Very flexible (13+ rot bonds): large divergence expected ---
    {
        "name": "hexadecane",
        "smiles": "CCCCCCCCCCCCCCCC",
        "rot_bonds": 13,
        "flexibility_class": "very_flexible",
        "notes": "C16 alkane, 13 rot bonds — Boltzmann tier 300 conformers",
    },
    {
        "name": "docosane",
        "smiles": "CCCCCCCCCCCCCCCCCCCCCC",
        "rot_bonds": 19,
        "flexibility_class": "very_flexible",
        "notes": "C22 alkane, 19 rot bonds — max conformational spread in this set",
    },
]

COLUMN_ORDER = ["id", "name", "smiles", "rot_bonds", "flexibility_class", "notes"]


def build_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(REFERENCE_COMPOUNDS)
    df.insert(0, "id", range(len(df)))
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMN_ORDER]


DESCRIPTION = {
    "description": (
        "Reference compounds spanning the rotatable-bond spectrum (0 → 19). "
        "Used by flexibility-comparison tests to surface divergence between "
        "the fast 3D endpoint (10 conformers) and the Boltzmann async endpoint "
        "(50-300 adaptive conformers). Rigid compounds serve as same-math "
        "regression gates; flexible compounds show how much more thorough "
        "conformer sampling actually shifts ensemble-averaged descriptors."
    ),
    "columns": {
        "id": "Integer row index",
        "name": "Short identifier",
        "smiles": "Canonical SMILES",
        "rot_bonds": "Rotatable-bond count (from RDKit CalcNumRotatableBonds)",
        "flexibility_class": "One of: rigid, semi_rigid, flexible, very_flexible",
        "notes": "Why this compound is in the set",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Populate the flexibility reference-compounds dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print the DataFrame without uploading")
    args = parser.parse_args()

    df = build_dataframe()
    DESCRIPTION["num_compounds"] = int(len(df))
    run_populate(df, CSV_KEY, DESCRIPTION, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
