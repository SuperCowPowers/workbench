from sklearn.neighbors import NearestNeighbors
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Group A (similar within group)
group_a_smiles = ["CCO", "CCOCC", "CCOCCC", "CCOC", "CCOCO"]

# Group B (similar within group, dissimilar to group A)
group_b_smiles = ["CCCCN", "CCCCCN", "CCCCCCN", "CCCCCCCN", "CCCCCCCCN"]

# Combine groups
smiles_list = group_a_smiles + group_b_smiles

# Generate fingerprints using MorganGenerator
fingerprints = []
for sm in smiles_list:
    mol = Chem.MolFromSmiles(sm)
    if mol:
        gen = rdMolDescriptors.MorganGenerator(radius=2, fpSize=2048)
        fingerprints.append(np.array(gen.GetFingerprint(mol), dtype=np.int8))

fingerprints_array = np.array(fingerprints)

# Fit NearestNeighbors (use n_neighbors < total samples)
n_neighbors = 3  # Adjust to fit dataset size
nn = NearestNeighbors(metric="jaccard", n_neighbors=n_neighbors)
nn.fit(fingerprints_array)

# Query for neighbors
query_fp = fingerprints_array[0]  # First molecule from group A
distances, indices = nn.kneighbors([query_fp])

# Convert Jaccard distances to Tanimoto similarities
tanimoto_similarities = 1 - distances

# Output results
print("Query molecule:", smiles_list[0])
print("Nearest neighbors (indices):", indices[0])
print("Nearest neighbors (SMILES):", [smiles_list[i] for i in indices[0]])
print("Tanimoto similarities:", tanimoto_similarities[0])
