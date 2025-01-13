# Workbench Imports
from workbench.api.df_store import DFStore
from workbench.utils.chem_utils import compute_morgan_fingerprints
from rdkit.Chem import DataStructs
from rdkit.DataStructs.cDataStructs import CreateFromBitString
import pandas as pd

# Grab tox21 dataset (just 10)
tox_df = DFStore().get("/datasets/chem_info/tox21_10")

# Compute Morgan fingerprints as bitstrings
tox_df = compute_morgan_fingerprints(tox_df)

# Extract fingerprints and IDs
fingerprint_column = "morgan_fingerprint"
id_column = "id"
fingerprints = tox_df[fingerprint_column].tolist()
compound_ids = tox_df[id_column].tolist()

# Convert bitstrings to ExplicitBitVect objects
explicit_fingerprints = [CreateFromBitString(fp) for fp in fingerprints]

# Compute Tanimoto similarities using RDKit
results = []
for i, query_fp in enumerate(explicit_fingerprints):
    # Compute bulk Tanimoto similarity
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, explicit_fingerprints)
    # Get top 5 neighbors (excluding the query itself)
    top_indices = sorted(range(len(similarities)), key=lambda j: -similarities[j])[:6]  # Include self
    top_neighbors = [
        {"query_id": compound_ids[i], "neighbor_id": compound_ids[j], "similarity": similarities[j]}
        for j in top_indices
        if compound_ids[j] != compound_ids[i]
    ][
        :5
    ]  # Exclude self and limit to top 5
    results.extend(top_neighbors)

# Convert results to a DataFrame
neighbors_df = pd.DataFrame(results)

# Display all neighbors for all compounds
print("\nAll neighbors:")
print(neighbors_df.head(20))

# Query for neighbors for a specific compound
query_id = tox_df[id_column].iloc[1]
query_neighbors_df = neighbors_df[neighbors_df["query_id"] == query_id]
print(f"\nBULK: Neighbors for query ID {query_id}:")
print(query_neighbors_df)
