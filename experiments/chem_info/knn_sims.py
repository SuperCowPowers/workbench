# Workbench Imports
from workbench.api.df_store import DFStore
from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

# Grab tox21 dataset (only 10)
tox_df = DFStore().get("/datasets/chem_info/tox21_10")
# tox_df = tox_df.sample(100)

# Compute FingerprintProximity (auto-computes fingerprints from SMILES)
prox = FingerprintProximity(tox_df, id_column="id")

# Get all neighbors for the first 3 compounds
all_ids = prox.df["id"].tolist()[:3]
neighbors_df = prox.neighbors(all_ids, n_neighbors=5)
print("\nNeighbors for first 3 compounds:")
print(neighbors_df.head(20))

# Query for neighbors for a specific compound
query_id = prox.df["id"].iloc[1]
query_neighbors_df = prox.neighbors(query_id, n_neighbors=5)
print(f"\nKNN: Neighbors for query ID {query_id}:")
print(query_neighbors_df)
