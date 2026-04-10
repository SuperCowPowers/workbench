"""Endpoint integration test for the 3D descriptor feature endpoint.

Sends a DataFrame of public compounds (including known corner cases) directly
to the deployed smiles-to-3d-descriptors endpoint and validates the response.

This tests the full deployed stack: SageMaker container, model script, standardization,
conformer generation, and descriptor computation.

Usage:
    python scripts/admin/test_3d_endpoint.py
    python scripts/admin/test_3d_endpoint.py smiles-to-3d-descriptors-v2
"""

import argparse
import pandas as pd
from workbench.api import Endpoint
from workbench.utils.chem_utils.mol_descriptors_3d import get_3d_feature_names

DEFAULT_ENDPOINT = "smiles-to-3d-descriptors-v1"

# All public compounds with known expected behavior
TEST_COMPOUNDS = pd.DataFrame(
    {
        "smiles": [
            # Should succeed — common public drugs
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
            "OB(O)c1ccccc1",  # Phenylboronic acid
            "CCCCCCCCCCCCCCCCCCCC",  # Eicosane
            # Should succeed after salt extraction
            "[Na+].CC(=O)[O-]",  # Sodium acetate
            "F[B-](F)(F)F.[K+]",  # Potassium tetrafluoroborate
            # Should be skipped (NaN) — complexity guard / problematic scaffolds
            "C1CC2CC1CC2O",  # Norbornanol — norbornane core
            "C12C3C4C1C5C3C2C45",  # Cubane — dense cage
            # Edge cases that should not crash the endpoint
            "",  # Empty SMILES
            "INVALID_SMILES",  # Invalid SMILES
            "[2H]C([2H])([2H])O",  # Deuterated methanol (CD3OD)
        ],
        "id": [
            "ethanol",
            "benzene",
            "aspirin",
            "caffeine",
            "testosterone",
            "ibuprofen",
            "phenylboronic_acid",
            "eicosane",
            "sodium_acetate",
            "potassium_tetrafluoroborate",
            "norbornanol",
            "cubane",
            "empty",
            "invalid",
            "deuterated_methanol",
        ],
    }
)

# Expected results: True = should have descriptor values, False = should be NaN
EXPECTED = {
    "ethanol": True,
    "benzene": True,
    "aspirin": True,
    "caffeine": True,
    "testosterone": True,
    "ibuprofen": True,
    "phenylboronic_acid": True,
    "eicosane": True,
    "sodium_acetate": True,
    "potassium_tetrafluoroborate": True,
    "norbornanol": False,  # Norbornane core — skipped by complexity check
    "cubane": False,  # Dense cage — skipped by complexity check
    "empty": False,  # No SMILES
    "invalid": False,  # Unparseable
    "deuterated_methanol": True,
}


def run_endpoint_test(endpoint_name: str = DEFAULT_ENDPOINT):
    """Send test compounds to the endpoint and validate results."""

    print(f"\n{'=' * 80}")
    print(f"Endpoint Integration Test — {endpoint_name}")
    print(f"{'=' * 80}")

    # Check endpoint exists and is ready
    end = Endpoint(endpoint_name)
    if not end.exists():
        print(f"ERROR: Endpoint '{endpoint_name}' does not exist")
        return False

    print(f"Sending {len(TEST_COMPOUNDS)} test compounds...\n")

    # Send to endpoint
    result = end.inference(TEST_COMPOUNDS)

    # Validate response shape
    feature_names = get_3d_feature_names()
    missing_cols = [f for f in feature_names if f not in result.columns]
    if missing_cols:
        print(f"FAIL: Missing {len(missing_cols)} feature columns: {missing_cols[:5]}...")
        return False
    print(f"Response has all {len(feature_names)} feature columns")

    # Validate each compound
    first_feature = feature_names[0]
    n_pass = 0
    n_fail = 0

    for _, row in result.iterrows():
        compound_id = row["id"]
        has_values = pd.notna(row[first_feature])
        expected = EXPECTED.get(compound_id)

        if expected is None:
            print(f"  [?] UNKNOWN  {compound_id} — not in expected results")
            continue

        if has_values == expected:
            status = "PASS"
            n_pass += 1
        else:
            status = "FAIL"
            n_fail += 1

        detail = f"got values={has_values}, expected={expected}"
        symbol = {"PASS": "+", "FAIL": "!"}[status]
        print(f"  [{symbol}] {status:4s}  {compound_id:30s}  {detail}")

    print(f"\n{'─' * 60}")
    print(f"Results: {n_pass} passed, {n_fail} failed out of {len(EXPECTED)}")
    print(f"{'─' * 60}")

    return n_fail == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the 3D descriptor endpoint")
    parser.add_argument("endpoint", nargs="?", default=DEFAULT_ENDPOINT, help="Endpoint name")
    args = parser.parse_args()
    success = run_endpoint_test(args.endpoint)
    exit(0 if success else 1)
