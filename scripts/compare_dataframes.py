from workbench.api import DFStore
from workbench.utils.pandas_utils import compare_dataframes

# Initialize DFStore and load dataframes
df_store = DFStore()
old_df = df_store.get("/smiles_testing/hlh/old_endpoint")
new_df = df_store.get("/smiles_testing/hlh/new_endpoint")

# Convert all columns names to lowercase
old_df.columns = old_df.columns.str.lower()

# We know that the old endpoint crashes on these two smiles (so the endpoint harness puts in NaNs)
old_error_smiles_1 = "CCC1C2=N3C(=CC4=C(C(C)=O)C(C)=C5C=C6C(C)C(CCC(=O)O)C7=N6[Pd+2]3([N-]54)[N-]3C(=C2)C(C)=C(C(=O)NCCS(=O)(=O)O)C3=C7CC(=O)OC)C1C"
old_error_smiles_2 = "O=C(CCC1C(=O)[O-][Gd+3]23456[O-]C(=O)C(CCC(=O)NCC(O)CO)N27CCN13CCN4(CC1=N5C(=CC=C1)C7)C(CCC(=O)NCC(O)CO)C(=O)[O-]6)NCC(O)CO"

# Remove this smiles from both dataframes
old_df = old_df[~old_df["smiles"].isin([old_error_smiles_1, old_error_smiles_2])]
new_df = new_df[~new_df["smiles"].isin([old_error_smiles_1, old_error_smiles_2])]

# Compare the dataframes
comparison = compare_dataframes(old_df, new_df, ["udm_mol_bat_id", "smiles"])
