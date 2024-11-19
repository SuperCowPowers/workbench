from sageworks.api import DFStore
from sageworks.utils.pandas_utils import compare_dataframes

# Initialize DFStore and load dataframes
df_store = DFStore()
current_df = df_store.get("/testing/current_features")
new_df = df_store.get("/testing/new_features")

# Compare the dataframes
comparison = compare_dataframes(current_df, new_df, ["udm_mol_bat_id", "SMILES"])
