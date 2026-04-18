"""Populate the public reference-compounds dataset for 3D complexity-skip tests.

Builds a curated table of SMILES that should each trigger a specific
desc3d_status skip path in compute_descriptors_3d — one compound per path,
plus an 'ok' control to make sure normal compounds aren't accidentally
caught by the guards.

Dataset location:
    s3://workbench-public-data/comp_chem/reference_compounds/complexity_skips.csv

Access (read-only, unsigned):
    from workbench.api import PublicData
    df = PublicData().get("comp_chem/reference_compounds/complexity_skips")

Consumed by:
    tests/feature_endpoints/test_3d_complexity_skips.py

Usage:
    python scripts/admin/populate_complexity_skip_reference_compounds.py --dry-run
    python scripts/admin/populate_complexity_skip_reference_compounds.py
"""

import argparse
import logging

import pandas as pd

from reference_compounds_common import run_populate

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

CSV_KEY = "comp_chem/reference_compounds/complexity_skips.csv"

# Expected statuses below were captured from the current complexity-check
# thresholds in mol_descriptors_3d.py:
#   MAX_HEAVY_ATOMS = 100, MAX_ROTATABLE_BONDS = 30,
#   MAX_RING_SYSTEMS = 10, MAX_RING_COMPLEXITY = 15
# Changes to those thresholds would shift these expected values.
REFERENCE_COMPOUNDS = [
    # Control — within all thresholds, should compute successfully
    {
        "name": "eicosanoic_acid",
        "smiles": "CCCCCCCCCCCCCCCCCCCC(=O)O",
        "expected_status": "ok",
        "notes": "20-carbon fatty acid (18 rot bonds) — just under the 30-rot-bond limit",
    },
    {
        "name": "cyclosporin_a",
        "smiles": (
            "CC[C@H]1NC(=O)[C@@H](CC(C)C)N(C)C(=O)[C@H](C(C)C)N(C)C(=O)[C@H](CC(C)C)"
            "N(C)C(=O)[C@H](C)N[C@H](C(=O)N[C@H](C(C)C)C(=O)N([C@H]([C@@H](C)[C@H](C)"
            "C/C=C/C)C(=O)N1C)C)[C@@H](C)O"
        ),
        "expected_status": "ok",
        "notes": (
            "Cyclic peptide — macrocycle passes all complexity checks (SSSR reports "
            "only 1 ring for the single macrocyclic loop). Guards intentionally do "
            "NOT block macrocycles; that's a separate design decision."
        ),
    },
    # skip:rot_bonds (>30 rotatable bonds)
    {
        "name": "linear_c40",
        "smiles": "C" * 40,
        "expected_status": "skip:rot_bonds",
        "notes": "C40 n-alkane with 37 rot bonds > 30",
    },
    # skip:heavy_atoms (>100 heavy atoms)
    {
        "name": "glycine_50mer",
        "smiles": "NCC(=O)" + "NCC(=O)" * 50 + "O",
        "expected_status": "skip:heavy_atoms",
        "notes": "51-residue poly-glycine (205 heavy atoms > 100)",
    },
    # skip:rings (>10 ring systems)
    {
        "name": "c60_fullerene",
        "smiles": ("c12c3c4c5c1c6c7c2c8c3c9c4c%10c5c%11c6c%12c7c8c%13c9c%10c%14c%11c%12c%13%14"),
        "expected_status": "skip:rings",
        "notes": "C60 fullerene — SSSR reports 15 rings > 10",
    },
    # skip:ring_complexity (rings + bridgehead + spiro > 15)
    {
        "name": "kekulene_like_polycycle",
        "smiles": "c1ccc2ccc3ccc4ccc5ccc6ccc7ccc1c8ccc2c3c8c4c5c6c7",
        "expected_status": "skip:ring_complexity",
        "notes": "Fused polycyclic — ring_complexity=17 (rings=9 + bridgehead=8) > 15",
    },
    # skip:embed (passes complexity guards, fails conformer gen)
    {
        "name": "cubane",
        "smiles": "C12C3C4C1C5C3C2C45",
        "expected_status": "skip:embed",
        "notes": "Dense strained cage — passes complexity check but ETKDGv3 can't embed",
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
        "Reference compounds for 3D complexity-guard tests. Each row pairs a "
        "SMILES with the expected value of desc3d_status produced by "
        "workbench.utils.chem_utils.mol_descriptors_3d.compute_descriptors_3d, "
        "designed to exercise each specific skip path (heavy_atoms, rot_bonds, "
        "rings, ring_complexity, embed) plus 'ok' controls."
    ),
    "columns": {
        "id": "Integer row index",
        "name": "Short identifier for the test case",
        "smiles": "SMILES fed to compute_descriptors_3d",
        "expected_status": (
            "Expected value of the desc3d_status diagnostic column. Possible "
            "values: 'ok', 'skip:heavy_atoms', 'skip:rot_bonds', 'skip:rings', "
            "'skip:ring_complexity', 'skip:embed'."
        ),
        "notes": "Why this compound exercises the named skip path",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Populate the complexity-skip reference-compounds dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print the DataFrame without uploading")
    args = parser.parse_args()

    df = build_dataframe()
    DESCRIPTION["num_compounds"] = int(len(df))
    run_populate(df, CSV_KEY, DESCRIPTION, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
