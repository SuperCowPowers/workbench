from workbench.utils.chem_utils import canonicalize
import pandas as pd


def test_canonicalize():
    """Test that canonicalize resolves known SMILES to their expected canonical forms."""
    # Define test cases for canonicalization
    test_data = [
        {"id": "Acetylacetone", "smiles": "CC(=O)CC(=O)C", "expected": "CC(=O)CC(C)=O"},
        {"id": "Imidazole", "smiles": "c1cnc[nH]1", "expected": "c1c[nH]cn1"},
        {"id": "Pyridone", "smiles": "C1=CC=NC(=O)C=C1", "expected": "O=c1cccccn1"},
        {"id": "Guanidine", "smiles": "C(=N)N=C(N)N", "expected": "N=CN=C(N)N"},
        {"id": "Catechol", "smiles": "c1cc(c(cc1)O)O", "expected": "Oc1ccccc1O"},
        {"id": "Formamide", "smiles": "C(=O)N", "expected": "NC=O"},
        {"id": "Urea", "smiles": "C(=O)(N)N", "expected": "NC(N)=O"},
        {"id": "Phenol", "smiles": "c1ccc(cc1)O", "expected": "Oc1ccccc1"},
        {"id": "Lactic Acid", "smiles": "CC(O)C(=O)O", "expected": "CC(O)C(=O)O"},  # Same as input
        {"id": "Pyruvic Acid", "smiles": "CC(=O)C(=O)O", "expected": "CC(=O)C(=O)O"},  # Same as input
        {"id": "Nitromethane", "smiles": "C[N+](=O)[O-]", "expected": "C[N+](=O)[O-]"},  # Same as input
    ]

    # Convert test data to a DataFrame
    df = pd.DataFrame(test_data)

    # Perform canonicalization
    result_df = canonicalize(df)

    # Test that the results match the expected canonical SMILES
    for index, row in result_df.iterrows():
        assert row["smiles_canonical"] == row["expected"], f"Canonicalization failed for {row['id']}"

    print("All canonicalization tests passed!")


if __name__ == "__main__":
    # Run the test
    test_canonicalize()
