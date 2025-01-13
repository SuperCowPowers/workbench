# Workbench Imports
from workbench.api.df_store import DFStore
from workbench.utils.chem_utils import compute_morgan_fingerprints
from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

# Grab tox21 dataset (only 10)
tox_df = DFStore().get("/datasets/chem_info/tox21_10")
# tox_df = tox_df.sample(100)

# Compute Morgan fingerprints
tox_df = compute_morgan_fingerprints(tox_df)

# Compute FingerprintProximity
prox = FingerprintProximity(tox_df, fingerprint_column="morgan_fingerprint", id_column="id", n_neighbors=5)

# Get all the neighbors for all the compounds
neighbors_df = prox.all_neighbors()
print("\nAll neighbors:")
print(neighbors_df.head(20))

# Query for neighbors for a specific compound
query_id = tox_df["id"].iloc[1]
query_neighbors_df = prox.neighbors(query_id=query_id)
print(f"\nKNN: Neighbors for query ID {query_id}:")
print(query_neighbors_df)
