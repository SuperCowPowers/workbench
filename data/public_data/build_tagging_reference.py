"""Build the public reference-compounds dataset for mol_tagging() validation.

Writes a curated table of input SMILES paired with the expected curation tags
produced by:

    workbench.utils.chem_utils.mol_standardize.standardize  ->
    workbench.utils.chem_utils.mol_tagging.tag_molecules

The dataset complements ``comp_chem/reference_compounds/standardization`` —
that set exercises salt / charge / tautomer / stereo handling, whereas this
set exercises ADMET-curation tag derivation across composition, structure,
physchem, liability, and curation categories.

Output:
    data/public_data/output/reference_compounds/tagging.csv

Publish:
    python data/public_data/build_tagging_reference.py
    AWS_PROFILE=scp_sandbox_admin python data/public_data/upload_data.py --apply
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from workbench.utils.chem_utils.mol_standardize import standardize
from workbench.utils.chem_utils.mol_tagging import tag_molecules

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "output" / "reference_compounds"
CSV_PATH = OUTPUT_DIR / "tagging.csv"


# ---------------------------------------------------------------------------
# Reference compounds
#
# Each entry pairs a raw input SMILES with the EXPECTED curation tags emitted
# by tag_molecules() after running the input through standardize() first.
# Coverage targets every curation tag the module can emit, plus a handful of
# clean drug controls that should carry no curation flags.
#
# Note: ``curation:exclude:mixture`` is not represented — after standardize()
# strips counterions, true post-standardization mixtures are extremely rare
# in practice. Inorganic salts (e.g. ZnCl2) typically reduce to a single
# metal-cation fragment, so they exercise ``exclude:inorganic`` instead.
# ---------------------------------------------------------------------------

REFERENCE_COMPOUNDS = [
    # --- exclude:inorganic --------------------------------------------------
    {
        "name": "water",
        "input_smiles": "O",
        "expected_curation_tags": [
            "curation:exclude:inorganic",
            "curation:exclude:mw_too_low",
        ],
        "notes": "No-carbon inorganic — exercises exclude:inorganic + exclude:mw_too_low",
    },
    {
        "name": "zinc_chloride",
        "input_smiles": "[Zn+2].[Cl-].[Cl-]",
        "expected_curation_tags": [
            "curation:exclude:inorganic",
            "curation:exclude:mw_too_low",
            "curation:caution:heavy_metal",
            "curation:caution:unusual_element",
        ],
        "notes": "Inorganic salt — standardize keeps [Zn+2] as parent; still inorganic + heavy metal",
    },
    # --- exclude:organometallic --------------------------------------------
    {
        "name": "tetraethyllead",
        "input_smiles": "CC[Pb](CC)(CC)CC",
        "expected_curation_tags": [
            "curation:exclude:organometallic",
            "curation:caution:heavy_metal",
            "curation:caution:unusual_element",
        ],
        "notes": "Covalent C-Pb organometallic — exercises exclude:organometallic",
    },
    # --- caution:peptide ---------------------------------------------------
    {
        "name": "tetra_alanine",
        "input_smiles": "N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)O",
        "expected_curation_tags": ["curation:caution:peptide"],
        "notes": "3 peptide bonds — at threshold for peptide caution",
    },
    # --- caution:macrocycle ------------------------------------------------
    {
        "name": "cyclododecane",
        "input_smiles": "C1CCCCCCCCCCC1",
        "expected_curation_tags": ["curation:caution:macrocycle"],
        "notes": "12-membered ring at IUPAC macrocycle boundary",
    },
    # --- caution:isotope_labeled -------------------------------------------
    {
        "name": "d3_methanol",
        "input_smiles": "[2H]C([2H])([2H])O",
        "expected_curation_tags": [
            "curation:caution:isotope_labeled",
            "curation:exclude:mw_too_low",
        ],
        "notes": "Deuterium labels — exercises isotope detection (also too small)",
    },
    {
        "name": "d3_naproxen",
        "input_smiles": "[2H]C([2H])([2H])[C@H](C(=O)O)c1ccc2cc(OC)ccc2c1",
        "expected_curation_tags": ["curation:caution:isotope_labeled"],
        "notes": "Drug-sized isotope-labeled compound (isotope flag without mw confound)",
    },
    # --- caution:stereo_undefined ------------------------------------------
    {
        "name": "alanine_unspecified",
        "input_smiles": "NC(C)C(=O)O",
        "expected_curation_tags": [
            "curation:caution:stereo_undefined",
            "curation:exclude:mw_too_low",
        ],
        "notes": "Has a chiral center with no stereo flag — features reflect arbitrary enantiomer",
    },
    # --- caution:highly_halogenated ---------------------------------------
    {
        "name": "ddt",
        "input_smiles": "ClC(Cl)(Cl)C(c1ccc(Cl)cc1)c2ccc(Cl)cc2",
        "expected_curation_tags": ["curation:caution:highly_halogenated"],
        "notes": "5 Cl atoms — exceeds size-scaled halogen threshold",
    },
    # --- caution:pains -----------------------------------------------------
    {
        "name": "catechol",
        "input_smiles": "c1ccc(O)c(O)c1",
        "expected_curation_tags": ["curation:caution:pains"],
        "notes": "Catechol substructure — known PAINS_B match (assay interference)",
    },
    # --- exclude:mw_too_high + multi-tag combo ----------------------------
    {
        "name": "polyalanine_13",
        "input_smiles": "N" + "C(C)C(=O)N" * 12 + "C(C)C(=O)O",
        "expected_curation_tags": [
            "curation:exclude:mw_too_high",
            "curation:caution:peptide",
            "curation:caution:stereo_undefined",
        ],
        "notes": "13-residue peptide (MW>900) — peptide + mw_too_high + undefined stereo",
    },
    # --- clean drug controls (no curation tags) ---------------------------
    {
        "name": "aspirin",
        "input_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "expected_curation_tags": [],
        "notes": "Small-molecule drug control — no curation flags expected",
    },
    {
        "name": "caffeine",
        "input_smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "expected_curation_tags": [],
        "notes": "Small-molecule drug control",
    },
    {
        "name": "paracetamol",
        "input_smiles": "CC(=O)Nc1ccc(O)cc1",
        "expected_curation_tags": [],
        "notes": "Small-molecule drug control",
    },
    {
        "name": "naproxen",
        "input_smiles": "C[C@H](C(=O)O)c1ccc2cc(OC)ccc2c1",
        "expected_curation_tags": [],
        "notes": "Chiral drug control with defined stereo — no stereo_undefined caution",
    },
    {
        "name": "ibuprofen_racemic",
        "input_smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "expected_curation_tags": ["curation:caution:stereo_undefined"],
        "notes": "Racemic drug (no stereo flag) — legitimately triggers stereo_undefined caution",
    },
]

COLUMN_ORDER = [
    "id",
    "name",
    "input_smiles",
    "standardized_smiles",
    "expected_undefined_chiral_centers",
    "expected_curation_tags",
    "notes",
]


def build_dataframe(verify: bool = True) -> pd.DataFrame:
    """Build the reference DataFrame. Runs the pipeline and captures the
    standardized SMILES + undefined-chiral count for each row; if ``verify``
    is True, also checks that the live curation tags match the declared
    ``expected_curation_tags`` and raises on any mismatch.
    """
    df = pd.DataFrame(REFERENCE_COMPOUNDS)
    df.insert(0, "id", range(len(df)))

    # Run the pipeline so we can capture the standardized SMILES and the
    # undefined-chiral-center count, and (optionally) verify the expected tags.
    pipe_df = df.rename(columns={"input_smiles": "smiles"})[["id", "name", "smiles"]].copy()
    std_df = standardize(pipe_df)
    tagged = tag_molecules(std_df)

    df["standardized_smiles"] = std_df["smiles"].values
    df["expected_undefined_chiral_centers"] = std_df["undefined_chiral_centers"].values

    if verify:
        mismatches = []
        for _, row in df.iterrows():
            actual = sorted(
                t for t in tagged.loc[row["id"], "tags"] if t.startswith("curation:")
            )
            expected = sorted(row["expected_curation_tags"])
            if actual != expected:
                mismatches.append((row["name"], expected, actual))
        if mismatches:
            log.error("Expected-tag mismatches:")
            for name, exp, act in mismatches:
                log.error(f"  {name}:")
                log.error(f"    expected: {exp}")
                log.error(f"    actual:   {act}")
            raise SystemExit(
                "Reference compounds expected_curation_tags out of sync with "
                "current tag_molecules() output. Fix the expected values or the "
                "tagging logic before publishing."
            )

    # Serialize the list column as pipe-delimited for CSV.
    df["expected_curation_tags"] = df["expected_curation_tags"].apply(lambda lst: "|".join(lst))

    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMN_ORDER]


def main():
    parser = argparse.ArgumentParser(description="Build the tagging reference-compounds CSV")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip live re-verification of expected_curation_tags",
    )
    args = parser.parse_args()

    df = build_dataframe(verify=not args.no_verify)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    log.info(f"Wrote {len(df)} rows -> {CSV_PATH}")
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.max_colwidth", 60):
        log.info("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
