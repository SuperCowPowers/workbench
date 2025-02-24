"""Example for Tautomerizing SMILES strings"""

import pandas as pd
from workbench.utils.chem_utils import tautomerize_smiles

pd.options.display.max_columns = None
pd.options.display.width = 1200


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
]

# Convert test data to a DataFrame
df = pd.DataFrame(test_data)

# Perform tautomerization
result_df = tautomerize_smiles(df)
print(result_df)
