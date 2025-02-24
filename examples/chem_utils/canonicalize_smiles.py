"""Example for computing Canonicalize SMILES strings"""

import pandas as pd
from workbench.utils.chem_utils import canonicalize

test_data = [
    {"id": "Acetylacetone", "smiles": "CC(=O)CC(=O)C", "expected": "CC(=O)CC(C)=O"},
    {"id": "Imidazole", "smiles": "c1cnc[nH]1", "expected": "c1c[nH]cn1"},
    {"id": "Pyridone", "smiles": "C1=CC=NC(=O)C=C1", "expected": "O=c1cccccn1"},
    {"id": "Guanidine", "smiles": "C(=N)N=C(N)N", "expected": "N=CN=C(N)N"},
    {"id": "Catechol", "smiles": "c1cc(c(cc1)O)O", "expected": "Oc1ccccc1O"},
    {"id": "Formamide", "smiles": "C(=O)N", "expected": "NC=O"},
    {"id": "Urea", "smiles": "C(=O)(N)N", "expected": "NC(N)=O"},
    {"id": "Phenol", "smiles": "c1ccc(cc1)O", "expected": "Oc1ccccc1"},
]

# Convert test data to a DataFrame
df = pd.DataFrame(test_data)

# Perform canonicalization
result_df = canonicalize(df)
print(result_df)
