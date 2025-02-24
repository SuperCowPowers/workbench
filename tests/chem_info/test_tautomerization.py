from workbench.utils.chem_utils import tautomerize_smiles


def test_tautomerization():
    """Test that tautomerization resolves known variants to the same canonical tautomer."""
    # Define known tautomeric variants
    test_data = [
        # Salicylaldehyde undergoes keto-enol tautomerization.
        {"id": "Salicylaldehyde (Keto)", "smiles": "O=Cc1cccc(O)c1", "expected": "O=Cc1cccc(O)c1"},
        {"id": "2-Hydroxybenzaldehyde (Enol)", "smiles": "Oc1ccc(C=O)cc1", "expected": "O=Cc1ccc(O)cc1"},
        # Acetylacetone undergoes keto-enol tautomerization to favor the enol form.
        {"id": "Acetylacetone", "smiles": "CC(=O)CC(=O)C", "expected": "CC(=O)CC(C)=O"},
        # Imidazole undergoes a proton shift in the aromatic ring.
        {"id": "Imidazole", "smiles": "c1cnc[nH]1", "expected": "c1c[nH]cn1"},
        # Pyridone prefers the lactam form in RDKit's tautomer enumeration.
        {"id": "Pyridone", "smiles": "C1=CC=NC(=O)C=C1", "expected": "O=c1cccccn1"},
        # Guanidine undergoes amine-imine tautomerization.
        {"id": "Guanidine", "smiles": "C(=N)N=C(N)N", "expected": "N=C(N)N=CN"},
        # Catechol standardizes hydroxyl group placement in the aromatic system.
        {"id": "Catechol", "smiles": "c1cc(c(cc1)O)O", "expected": "Oc1ccccc1O"},
        # Formamide canonicalizes to NC=O, reflecting its stable form.
        {"id": "Formamide", "smiles": "C(=O)N", "expected": "NC=O"},
        # Urea undergoes a proton shift between nitrogen atoms.
        {"id": "Urea", "smiles": "C(=O)(N)N", "expected": "NC(N)=O"},
        # Phenol standardizes hydroxyl group placement in the aromatic system.
        {"id": "Phenol", "smiles": "c1ccc(cc1)O", "expected": "Oc1ccccc1"},
        # Acetic Acid remains unchanged as it is already in a stable form.
        {"id": "Acetic Acid", "smiles": "CC(=O)O", "expected": "CC(=O)O"},
        # Nitromethane remains unchanged as it is already in a stable form.
        {"id": "Nitromethane", "smiles": "C[N+](=O)[O-]", "expected": "C[N+](=O)[O-]"},
        # Pyruvic Acid remains unchanged as it is already in a stable form.
        {"id": "Pyruvic Acid", "smiles": "CC(=O)C(=O)O", "expected": "CC(=O)C(=O)O"},
    ]

    # Convert test data to a DataFrame
    import pandas as pd

    df = pd.DataFrame(test_data)

    # Perform tautomerization
    result_df = tautomerize_smiles(df)

    # Test that the results match the expected canonical tautomers
    for index, row in result_df.iterrows():
        assert row["smiles"] == row["expected"], f"Tautomerization failed for {row['id']}"

    print("All tests passed!")


if __name__ == "__main__":

    # Run the tests
    test_tautomerization()
