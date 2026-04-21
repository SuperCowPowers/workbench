"""Populate the public reference-compounds dataset used by 3D endpoint
performance profiling.

Builds a table of 10 public drug analogs spanning the complexity spectrum
encountered by the 3D fast endpoint — from small NSAIDs to large HIV
protease inhibitors. Chosen to match the profile of pharma-scale inputs:
drug-like heterocycles, multiple stereo centers, and bridged bicyclic
amines (the structural feature that stresses ETKDGv3 embedding most).

Dataset location:
    s3://workbench-public-data/comp_chem/reference_compounds/3d_perf.csv

Access (read-only, unsigned):
    from workbench.api import PublicData
    df = PublicData().get("comp_chem/reference_compounds/3d_perf")

Consumed by:
    scripts/admin/profile_3d_fast.py

Usage:
    python scripts/admin/populate_3d_perf_reference_compounds.py --dry-run
    python scripts/admin/populate_3d_perf_reference_compounds.py
"""

import argparse
import logging

import pandas as pd

from reference_compounds_common import run_populate

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

CSV_KEY = "comp_chem/reference_compounds/3d_perf.csv"

# Public drug analogs spanning the complexity range seen by the 3D fast
# endpoint. Canonical SMILES from public sources (DrugBank / ChEMBL).
# Coverage bias is toward bridged bicyclics and multi-stereo scaffolds — the
# features that slow ETKDGv3 embedding on pharma-scale inputs.
REFERENCE_COMPOUNDS = [
    # Small NSAID — baseline for simple, low-stereo inputs
    {
        "name": "ibuprofen",
        "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "heavy_atoms": 15,
        "rot_bonds": 4,
        "bridgeheads": 0,
        "stereo_centers": 0,
        "notes": "Ibuprofen — NSAID, small drug-like baseline",
    },
    # Small rigid bridged bicyclic — nACh agonist
    {
        "name": "varenicline",
        "smiles": "c1cc2cc3c(cc2cn1)[C@H]4CN[C@@H]3CC4",
        "heavy_atoms": 16,
        "rot_bonds": 0,
        "bridgeheads": 2,
        "stereo_centers": 2,
        "notes": "Varenicline — nACh partial agonist, methano-bridged azabicycle",
    },
    # Tropane (bicyclo[2.2.1]) + ester — 5-HT3 antagonist
    {
        "name": "tropisetron",
        "smiles": "CN1[C@@H]2CC[C@H]1C[C@H](C2)OC(=O)c1c[nH]c2ccccc12",
        "heavy_atoms": 21,
        "rot_bonds": 2,
        "bridgeheads": 2,
        "stereo_centers": 3,
        "notes": "Tropisetron — 5-HT3 antagonist, tropane ester",
    },
    # Quinuclidine (bicyclo[2.2.2]) + stereo — M3 antagonist
    {
        "name": "solifenacin",
        "smiles": "O=C(O[C@H]1CN2CCC1CC2)[C@H]1c2ccccc2CCN1",
        "heavy_atoms": 21,
        "rot_bonds": 2,
        "bridgeheads": 2,
        "stereo_centers": 2,
        "notes": "Solifenacin — M3 antagonist, quinuclidine bridge",
    },
    # Tropane + epoxide + multi-stereo — muscarinic antagonist
    {
        "name": "scopolamine",
        "smiles": "CN1[C@H]2C[C@H](C[C@@H]1[C@@H]1[C@H]2O1)OC(=O)[C@@H](CO)c1ccccc1",
        "heavy_atoms": 22,
        "rot_bonds": 4,
        "bridgeheads": 2,
        "stereo_centers": 6,
        "notes": "Scopolamine — muscarinic antagonist, tropane + epoxide (6 stereo)",
    },
    # Small JAK inhibitor with one stereo center
    {
        "name": "ruxolitinib",
        "smiles": "N#CC[C@@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1",
        "heavy_atoms": 23,
        "rot_bonds": 4,
        "bridgeheads": 0,
        "stereo_centers": 1,
        "notes": "Ruxolitinib — JAK1/2 inhibitor, fused heteroaromatic",
    },
    # DPP-4 inhibitor with fused triazole
    {
        "name": "sitagliptin",
        "smiles": "N[C@@H](CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F",
        "heavy_atoms": 28,
        "rot_bonds": 4,
        "bridgeheads": 0,
        "stereo_centers": 1,
        "notes": "Sitagliptin — DPP-4 inhibitor, fluorinated triazole",
    },
    # EGFR inhibitor with flexible ether sidechains
    {
        "name": "erlotinib",
        "smiles": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        "heavy_atoms": 29,
        "rot_bonds": 10,
        "bridgeheads": 0,
        "stereo_centers": 0,
        "notes": "Erlotinib — EGFR kinase inhibitor, high rot-bond count",
    },
    # ALK inhibitor with halogenated aryl and piperidine
    {
        "name": "crizotinib",
        "smiles": "C[C@H](Oc1cc(-c2cnn(C3CCNCC3)c2)cnc1N)c1c(Cl)ccc(F)c1Cl",
        "heavy_atoms": 30,
        "rot_bonds": 5,
        "bridgeheads": 0,
        "stereo_centers": 1,
        "notes": "Crizotinib — ALK/ROS1 inhibitor, dichloro-fluoro aryl",
    },
    # CDK4/6 inhibitor with multiple fused rings
    {
        "name": "palbociclib",
        "smiles": "CC(=O)c1c(C)c2cnc(Nc3ccc(N4CCNCC4)nc3)nc2n(C3CCCC3)c1=O",
        "heavy_atoms": 33,
        "rot_bonds": 5,
        "bridgeheads": 0,
        "stereo_centers": 0,
        "notes": "Palbociclib — CDK4/6 inhibitor, pyrido-pyrimidinone fused",
    },
    # BTK inhibitor with complex fused pyrazolo-pyrimidine
    {
        "name": "ibrutinib",
        "smiles": "C=CC(=O)N1CCC[C@@H](C1)n1nc(-c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21",
        "heavy_atoms": 33,
        "rot_bonds": 5,
        "bridgeheads": 0,
        "stereo_centers": 1,
        "notes": "Ibrutinib — BTK inhibitor, acrylamide warhead + biaryl ether",
    },
    # BCR-ABL inhibitor, biaryl scaffold
    {
        "name": "imatinib",
        "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
        "heavy_atoms": 37,
        "rot_bonds": 7,
        "bridgeheads": 0,
        "stereo_centers": 0,
        "notes": "Imatinib — BCR-ABL kinase inhibitor, methylpiperazine biaryl",
    },
    # CCR5 antagonist — BRIDGED BICYCLIC (tropane); best analog for
    # the proprietary compound class that prompted this perf review.
    {
        "name": "maraviroc",
        "smiles": "CC(C)c1nnc(C)n1[C@H]1C[C@@H]2CC[C@H](C1)N2CC[C@@H](NC(=O)C1CCC(F)(F)CC1)c1ccc(F)cc1",
        "heavy_atoms": 38,
        "rot_bonds": 8,
        "bridgeheads": 2,
        "stereo_centers": 4,
        "notes": "Maraviroc — CCR5 antagonist, tropane bridge (2 bridgeheads)",
    },
    # HIV protease inhibitor — stress test for very large, flexible inputs
    {
        "name": "ritonavir",
        "smiles": (
            "CC(C)c1nc(CN(C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](Cc2ccccc2)"
            "C[C@H](O)[C@@H](Cc3ccccc3)NC(=O)OCc4cncs4)cs1"
        ),
        "heavy_atoms": 50,
        "rot_bonds": 17,
        "bridgeheads": 0,
        "stereo_centers": 4,
        "notes": "Ritonavir — HIV protease inhibitor, large multi-stereo stress test",
    },
    # HCV protease inhibitor — macrocyclic, bridged, 6 stereo — closest match
    # to internal compound complexity (large + bridged + multi-stereo)
    {
        "name": "simeprevir",
        "smiles": (
            "C[C@@H]1C[C@@H]2CCC(N[C@H]1C(=O)N1C[C@H](Oc3nc4cc(OC)c(C)cc4"
            "nc3-c3ccc(C(=O)NS(=O)(=O)C4CC4)cc3)C[C@H]1C(=O)N[C@]1(C(=O)"
            "N[C@@H]2C=C)CC1)=O"
        ),
        "heavy_atoms": 58,
        "rot_bonds": 8,
        "bridgeheads": 2,
        "stereo_centers": 6,
        "notes": "Simeprevir — HCV protease inhibitor, macrocyclic multi-stereo stress test",
    },
]

COLUMN_ORDER = ["id", "name", "smiles", "heavy_atoms", "rot_bonds", "bridgeheads", "stereo_centers", "notes"]


def build_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(REFERENCE_COMPOUNDS)
    df.insert(0, "id", range(len(df)))
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMN_ORDER]


DESCRIPTION = {
    "description": (
        "Public drug analogs used for 3D fast-endpoint performance profiling. "
        "Ten compounds spanning the complexity spectrum: small NSAID baseline "
        "through large HIV protease inhibitor. Includes a bridged-bicyclic "
        "amine (maraviroc/tropane) that stresses ETKDGv3 embedding the same "
        "way many pharma-scale inputs do. Consumed by profile_3d_fast.py to "
        "measure per-stage wall-clock (embed, optimize, energy, descriptor "
        "families) on representative inputs."
    ),
    "columns": {
        "id": "Integer row index",
        "name": "Public drug name (lowercase)",
        "smiles": "Canonical SMILES",
        "heavy_atoms": "Heavy atom count",
        "rot_bonds": "Rotatable bonds (RDKit CalcNumRotatableBonds)",
        "bridgeheads": "Bridgehead atom count (RDKit CalcNumBridgeheadAtoms)",
        "stereo_centers": "Specified chiral centers (no undefined)",
        "notes": "Public name and selection rationale",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Populate the 3D perf reference-compounds dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print the DataFrame without uploading")
    args = parser.parse_args()

    df = build_dataframe()
    DESCRIPTION["num_compounds"] = int(len(df))
    run_populate(df, CSV_KEY, DESCRIPTION, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
