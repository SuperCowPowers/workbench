"""Populate the public reference-compounds dataset used by 3D endpoint validation tests.

Builds a small, curated table of public compounds with known/expected 3D descriptor
values (shape class, IMHB count, molecular axis, nitrogen span, stereo preservation)
and uploads it to the workbench public S3 bucket so tests can pull it via PublicData().

Dataset location:
    s3://workbench-public-data/comp_chem/reference_compounds/reference_compounds_3d.csv

Access (read-only, unsigned):
    from workbench.api import PublicData
    df = PublicData().get("comp_chem/reference_compounds/reference_compounds_3d")

Consumed by:
    tests/feature_endpoints/test_rdkit_3d_v1.py
    tests/feature_endpoints/test_rdkit_3d_boltzmann_v1.py
    tests/feature_endpoints/test_3d_consistency.py

Usage:
    python scripts/admin/populate_3d_reference_compounds.py --dry-run
    python scripts/admin/populate_3d_reference_compounds.py
"""

import argparse
import logging

import pandas as pd

from reference_compounds_common import run_populate

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

CSV_KEY = "comp_chem/reference_compounds/reference_compounds_3d.csv"

# Reference compounds with known expected 3D-descriptor properties.
#
# Expected ranges are intentionally wide — the purpose is to catch gross errors
# (wrong sign, zero/NaN where values are expected, numbers off by orders of
# magnitude), not to nitpick numeric accuracy. Conformer-dependent descriptors
# can shift by 10-20% across runs with different random seeds, and the
# Boltzmann-weighted ensemble average can differ from the single lowest-energy
# value, so tolerances accommodate both modes.
#
# A range left as None skips that assertion for the compound.
#
# Geometric invariants (PMI ordering, NPR bounds, positivity) are checked
# for every compound regardless of range columns.
REFERENCE_COMPOUNDS = [
    # Shape classification on the NPR plot:
    #   rod:    (NPR1 ≈ 0,    NPR2 ≈ 1)
    #   disc:   (NPR1 ≈ 0.5,  NPR2 ≈ 0.5)
    #   sphere: (NPR1 ≈ 1,    NPR2 ≈ 1)
    {
        "name": "benzene",
        "smiles": "c1ccccc1",
        "shape_class": "disc",
        "npr1_min": 0.40,
        "npr1_max": 0.60,
        "npr2_min": 0.40,
        "npr2_max": 0.60,
        "imhb_min": 0.0,
        "imhb_max": 0.0,
        "axis_min": 2.4,
        "axis_max": 3.2,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Flat symmetric aromatic — textbook disc shape, no polar groups",
    },
    {
        "name": "naphthalene",
        "smiles": "c1ccc2ccccc2c1",
        "shape_class": "disc",
        "npr1_min": 0.20,
        "npr1_max": 0.45,
        "npr2_min": 0.55,
        "npr2_max": 0.80,
        "imhb_min": 0.0,
        "imhb_max": 0.0,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Planar bicyclic aromatic, elongated — off-corner disc",
    },
    {
        "name": "anthracene",
        "smiles": "c1ccc2cc3ccccc3cc2c1",
        "shape_class": "disc",
        "npr1_min": 0.10,
        "npr1_max": 0.30,
        "npr2_min": 0.70,
        "npr2_max": 0.90,
        "imhb_min": 0.0,
        "imhb_max": 0.0,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Three-ring linear acene — stretches NPR range further toward rod-like",
    },
    {
        "name": "decane",
        "smiles": "CCCCCCCCCC",
        "shape_class": "rod",
        "npr1_min": 0.00,
        "npr1_max": 0.25,
        "npr2_min": 0.80,
        "npr2_max": 1.01,
        "imhb_min": 0.0,
        "imhb_max": 0.0,
        "axis_min": 9.5,
        "axis_max": 13.0,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Extended all-trans alkane chain — textbook rod shape",
    },
    {
        "name": "adamantane",
        "smiles": "C1C2CC3CC1CC(C2)C3",
        "shape_class": "sphere",
        "npr1_min": 0.90,
        "npr1_max": 1.01,
        "npr2_min": 0.90,
        "npr2_max": 1.01,
        "imhb_min": 0.0,
        "imhb_max": 0.0,
        "undefined_chiral_centers_expected": 4,
        "stereo_preserved_expected": True,
        "notes": (
            "T_d-symmetric diamondoid cage — textbook sphere shape. "
            "RDKit's CIP detector flags the 4 bridgehead carbons as "
            "'potentially chiral' even though T_d symmetry makes them "
            "equivalent (not truly stereogenic). Expected behavior — the "
            "undefined_chiral_centers metric reports what RDKit detects, "
            "which is a useful upper bound for input-ambiguity warnings."
        ),
    },
    {
        "name": "biphenyl",
        "smiles": "c1ccc(-c2ccccc2)cc1",
        "shape_class": "",
        "imhb_min": 0.0,
        "imhb_max": 0.0,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Two aromatic rings with rotational freedom — twisted-disc torsion case",
    },
    # Intramolecular H-bond formers (ortho-OH next to H-bond acceptor)
    {
        "name": "salicylic_acid",
        "smiles": "O=C(O)c1ccccc1O",
        "shape_class": "",
        "imhb_min": 0.3,
        "imhb_max": 3.0,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Canonical IMHB former: ortho-OH donates to COOH carbonyl",
    },
    {
        "name": "2_hydroxybenzaldehyde",
        "smiles": "O=Cc1ccccc1O",
        "shape_class": "",
        "imhb_min": 0.3,
        "imhb_max": 3.0,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Canonical IMHB former: ortho-OH donates to aldehyde O",
    },
    # Nitrogen span (distance between furthest-apart N atoms)
    {
        "name": "ethylenediamine",
        "smiles": "NCCN",
        "shape_class": "",
        "nitrogen_span_min": 2.8,
        "nitrogen_span_max": 4.0,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "H2N-CH2-CH2-NH2, 4-bond N-to-N — short aliphatic diamine",
    },
    {
        "name": "p_phenylenediamine",
        "smiles": "Nc1ccc(N)cc1",
        "shape_class": "",
        "nitrogen_span_min": 5.3,
        "nitrogen_span_max": 6.0,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Para-diamine on benzene — medium N-span (ring diameter + 2× C-N)",
    },
    # Heteroaromatics / halogens (exercise amphiphilic_moment, HBA detection)
    {
        "name": "pyridine",
        "smiles": "c1ccncc1",
        "shape_class": "",
        "imhb_min": 0.0,
        "imhb_max": 0.0,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Aromatic N as pure H-bond acceptor — exercises hba_centroid_distance",
    },
    {
        "name": "nitrobenzene",
        "smiles": "[O-][N+](=O)c1ccccc1",
        "shape_class": "",
        "imhb_min": 0.0,
        "imhb_max": 0.0,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": (
            "Regression gate: nitro N/O must NOT be counted as HBA. Expected "
            "pharm3d_hba_centroid_dist=0 (the nitro group is the only "
            "heteroatom and is excluded from the acceptor filter)."
        ),
    },
    {
        "name": "tetramethylammonium",
        "smiles": "C[N+](C)(C)C",
        "shape_class": "",
        "imhb_min": 0.0,
        "imhb_max": 0.0,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": (
            "Regression gate: quaternary N+ must be counted as a charge site "
            "(formal_charge > 0 clause) but NOT as an HBA (no lone pair "
            "available). Td-symmetric — charge centroid coincides with COM."
        ),
    },
    {
        "name": "morpholine",
        "smiles": "C1COCCN1",
        "shape_class": "",
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Aliphatic heterocycle with both O acceptor and N-H donor in a ring",
    },
    {
        "name": "fluorobenzene",
        "smiles": "Fc1ccccc1",
        "shape_class": "",
        "imhb_min": 0.0,
        "imhb_max": 0.0,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Halogen + aromatic — exercises amphiphilic_moment C-F classification",
    },
    # Stereochemistry exercises
    {
        "name": "l_alanine",
        "smiles": "N[C@@H](C)C(=O)O",
        "shape_class": "",
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Smallest chiral reference — tests stereo preservation round-trip",
    },
    {
        "name": "s_ibuprofen",
        "smiles": "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1",
        "shape_class": "",
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Real drug with explicit stereo — tests stereo preservation on larger molecule",
    },
    {
        "name": "cis_stilbene",
        "smiles": "C(=C\\c1ccccc1)\\c1ccccc1",
        "shape_class": "",
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "E/Z bond stereo (complements sp3 stereo checks elsewhere)",
    },
    # General drug-like sanity checks (all features finite, status=ok)
    {
        "name": "ethanol",
        "smiles": "CCO",
        "shape_class": "",
        "imhb_min": 0.0,
        "imhb_max": 0.0,
        "axis_min": 1.8,
        "axis_max": 3.5,
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Smallest drug-like molecule with a polar group",
    },
    {
        "name": "caffeine",
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "shape_class": "",
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Rigid planar drug — general sanity check",
    },
    {
        "name": "aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "shape_class": "",
        "undefined_chiral_centers_expected": 0,
        "stereo_preserved_expected": True,
        "notes": "Standard drug-like reference",
    },
    {
        "name": "testosterone",
        "smiles": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
        "shape_class": "",
        "undefined_chiral_centers_expected": 6,
        "stereo_preserved_expected": True,
        "notes": "Steroid SMILES with no stereo markers — exercises undefined_chiral_centers=6 reporting",
    },
]

# Column order in the published CSV
COLUMN_ORDER = [
    "id",
    "name",
    "smiles",
    "shape_class",
    "npr1_min",
    "npr1_max",
    "npr2_min",
    "npr2_max",
    "imhb_min",
    "imhb_max",
    "axis_min",
    "axis_max",
    "nitrogen_span_min",
    "nitrogen_span_max",
    "undefined_chiral_centers_expected",
    "stereo_preserved_expected",
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
        "Curated public compounds with known expected 3D molecular descriptor "
        "values. Used by the workbench feature_endpoints tests to validate the "
        "3D descriptor pipeline (smiles-to-3d-descriptors-v1, "
        "smiles-to-3d-boltzmann-v1). Expected ranges are intentionally wide — "
        "the purpose is to catch gross errors, not nitpick numeric accuracy."
    ),
    "columns": {
        "id": "Integer row index (0 to N-1)",
        "name": "Compound common name",
        "smiles": "Canonical SMILES",
        "shape_class": "Expected NPR-plot shape — 'rod', 'disc', 'sphere', or empty",
        "npr1_min": "Expected lower bound on npr1 (null = not asserted)",
        "npr1_max": "Expected upper bound on npr1",
        "npr2_min": "Expected lower bound on npr2",
        "npr2_max": "Expected upper bound on npr2",
        "imhb_min": "Expected lower bound on pharm3d_imhb_potential (count)",
        "imhb_max": "Expected upper bound on pharm3d_imhb_potential",
        "axis_min": "Expected lower bound on pharm3d_molecular_axis (Angstroms, heavy atoms)",
        "axis_max": "Expected upper bound on pharm3d_molecular_axis",
        "nitrogen_span_min": "Expected lower bound on pharm3d_nitrogen_span (Angstroms)",
        "nitrogen_span_max": "Expected upper bound on pharm3d_nitrogen_span",
        "undefined_chiral_centers_expected": (
            "Expected value of the undefined_chiral_centers column produced by "
            "standardize() — count of chiral centers in the ORIGINAL input SMILES "
            "that have no stereo flag"
        ),
        "stereo_preserved_expected": (
            "Expected value of desc3d_stereo_preserved — True if the input's "
            "assigned stereo should survive 3D embedding (True for compounds "
            "with no chiral centers as well)"
        ),
        "notes": "Why this compound is in the reference table",
    },
    "references": [
        "https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors3D.html",
        "https://doi.org/10.1021/acs.jmedchem.5b00911 (IMHB / chameleonicity)",
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Populate the 3D reference-compounds public dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print the DataFrame without uploading")
    args = parser.parse_args()

    df = build_dataframe()
    DESCRIPTION["num_compounds"] = int(len(df))
    run_populate(df, CSV_KEY, DESCRIPTION, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
