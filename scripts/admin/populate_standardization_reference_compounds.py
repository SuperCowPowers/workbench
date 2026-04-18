"""Populate the public reference-compounds dataset for standardize() validation tests.

Builds a curated table of input SMILES paired with the expected outputs of
workbench.utils.chem_utils.mol_standardize.standardize() — the standardized
SMILES, the extracted salt (if any), the count of undefined chiral centers,
and whether the input should trigger the mixture-detected warning.

Dataset location:
    s3://workbench-public-data/comp_chem/reference_compounds/standardization.csv

Access (read-only, unsigned):
    from workbench.api import PublicData
    df = PublicData().get("comp_chem/reference_compounds/standardization")

Consumed by:
    tests/chem_info/test_standardization_reference.py

Usage:
    python scripts/admin/populate_standardization_reference_compounds.py --dry-run
    python scripts/admin/populate_standardization_reference_compounds.py
"""

import argparse
import logging

import pandas as pd

from reference_compounds_common import run_populate

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

CSV_KEY = "comp_chem/reference_compounds/standardization.csv"

# Expected values below were captured from the current standardize() pipeline
# (see scripts/admin/populate_standardization_reference_compounds.py docstring
# and the populate_standardization probe). Changes that cause any of these
# outputs to shift are either intentional pipeline changes (update the
# expected value + bump related downstream artifacts) or regressions.
REFERENCE_COMPOUNDS = [
    # --- Salt extraction ---
    {
        "name": "sodium_acetate",
        "input_smiles": "[Na+].CC(=O)[O-]",
        "expected_smiles": "CC(=O)O",
        "expected_salt": "[Na+]",
        "expected_undefined_chiral_centers": 0,
        "notes": "Canonical ionic salt — Na+ stripped, acetate neutralized to acetic acid",
    },
    {
        "name": "isoproterenol_hcl",
        "input_smiles": "CC(C)NCC(O)c1ccc(O)c(O)c1.Cl",
        "expected_smiles": "CC(C)NCC(O)c1ccc(O)c(O)c1",
        "expected_salt": "Cl",
        "expected_undefined_chiral_centers": 1,
        "notes": "Drug as HCl salt + undefined stereo on the benzylic alcohol",
    },
    {
        "name": "acid_base_mixture",
        "input_smiles": "CC(=O)O.CCN",
        "expected_smiles": "CCN",
        "expected_salt": "CC(=O)O",
        "expected_undefined_chiral_centers": 0,
        "notes": "Equal-size fragments — acetic acid is in KNOWN_COUNTERIONS so it's stripped",
    },
    # --- Charge neutralization ---
    {
        "name": "phenolate",
        "input_smiles": "[O-]c1ccccc1",
        "expected_smiles": "Oc1ccccc1",
        "expected_salt": None,
        "expected_undefined_chiral_centers": 0,
        "notes": "Charge neutralization: phenolate → phenol",
    },
    # --- Tautomer canonicalization ---
    {
        "name": "2_hydroxypyridine",
        "input_smiles": "Oc1ccccn1",
        "expected_smiles": "O=c1cccc[nH]1",
        "expected_salt": None,
        "expected_undefined_chiral_centers": 0,
        "notes": "Classic hydroxy↔oxo tautomer — canonicalizer prefers the pyridone form",
    },
    {
        "name": "imidazole",
        "input_smiles": "c1cnc[nH]1",
        "expected_smiles": "c1c[nH]cn1",
        "expected_salt": None,
        "expected_undefined_chiral_centers": 0,
        "notes": "Aromatic N-H tautomer canonicalization",
    },
    {
        "name": "acetylacetone",
        "input_smiles": "CC(=O)CC(=O)C",
        "expected_smiles": "CC(=O)CC(C)=O",
        "expected_salt": None,
        "expected_undefined_chiral_centers": 0,
        "notes": "Keto-enol — RDKit canonicalizer keeps the diketo form",
    },
    # --- Stereo preservation (critical: tautomerRemoveSp3Stereo=False) ---
    {
        "name": "l_alanine",
        "input_smiles": "N[C@@H](C)C(=O)O",
        "expected_smiles": "C[C@H](N)C(=O)O",
        "expected_salt": None,
        "expected_undefined_chiral_centers": 0,
        "notes": "sp3 stereo must survive tautomer canonicalization (regression gate)",
    },
    {
        "name": "l_valine",
        "input_smiles": "CC(C)[C@@H](N)C(=O)O",
        "expected_smiles": "CC(C)[C@@H](N)C(=O)O",
        "expected_salt": None,
        "expected_undefined_chiral_centers": 0,
        "notes": "sp3 stereo preservation — bulkier chiral amino acid",
    },
    {
        "name": "alanine_undefined",
        "input_smiles": "NC(C)C(=O)O",
        "expected_smiles": "CC(N)C(=O)O",
        "expected_salt": None,
        "expected_undefined_chiral_centers": 1,
        "notes": "Undefined-stereo counterpart to l_alanine — must report 1 center",
    },
    {
        "name": "trans_butene",
        "input_smiles": "C/C=C/C",
        "expected_smiles": "C/C=C/C",
        "expected_salt": None,
        "expected_undefined_chiral_centers": 0,
        "notes": "E/Z bond stereo preservation (tautomerRemoveBondStereo=False)",
    },
    {
        "name": "cis_butene",
        "input_smiles": "C/C=C\\C",
        "expected_smiles": "C/C=C\\C",
        "expected_salt": None,
        "expected_undefined_chiral_centers": 0,
        "notes": "E/Z bond stereo preservation — cis counterpart to trans_butene",
    },
    # --- Isotopes / no-op / corner cases ---
    {
        "name": "deuterated_methanol",
        "input_smiles": "[2H]C([2H])([2H])O",
        "expected_smiles": "[2H]C([2H])([2H])O",
        "expected_salt": None,
        "expected_undefined_chiral_centers": 0,
        "notes": "Deuterium labels preserved through standardization",
    },
    {
        "name": "benzene_no_op",
        "input_smiles": "c1ccccc1",
        "expected_smiles": "c1ccccc1",
        "expected_salt": None,
        "expected_undefined_chiral_centers": 0,
        "notes": "No-op control — benzene is already canonical; no changes expected",
    },
]

COLUMN_ORDER = [
    "id",
    "name",
    "input_smiles",
    "expected_smiles",
    "expected_salt",
    "expected_undefined_chiral_centers",
    "notes",
]


def build_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(REFERENCE_COMPOUNDS)
    df.insert(0, "id", range(len(df)))
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMN_ORDER]


DESCRIPTION = {
    "description": (
        "Reference compounds with known expected outputs of "
        "workbench.utils.chem_utils.mol_standardize.standardize(). Each row "
        "pairs an input SMILES with the expected standardized SMILES, "
        "extracted salt, and undefined-chiral-center count. Used as a "
        "regression gate — changes that shift any of these values are "
        "either intentional pipeline updates or bugs."
    ),
    "columns": {
        "id": "Integer row index",
        "name": "Short identifier for the test case",
        "input_smiles": "SMILES fed to standardize()",
        "expected_smiles": (
            "Expected value of the 'smiles' column in standardize() output "
            "(canonical form after cleanup + salt removal + charge "
            "neutralization + tautomer canonicalization)"
        ),
        "expected_salt": (
            "Expected value of the 'salt' column in standardize() output; "
            "empty (None in Python, empty string in CSV) means no salt extracted"
        ),
        "expected_undefined_chiral_centers": (
            "Expected value of the 'undefined_chiral_centers' column — count "
            "of chiral centers in the ORIGINAL input SMILES with no stereo flag"
        ),
        "notes": "Why this test case is in the reference set",
    },
    "references": [
        "https://doi.org/10.1186/s13321-020-00456-1 (ChEMBL structure standardization pipeline)",
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Populate the standardization reference-compounds dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print the DataFrame without uploading")
    args = parser.parse_args()

    df = build_dataframe()
    DESCRIPTION["num_compounds"] = int(len(df))
    run_populate(df, CSV_KEY, DESCRIPTION, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
