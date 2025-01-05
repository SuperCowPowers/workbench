from rdkit import Chem
from workbench.api.df_store import DFStore
from workbench.utils.chem_utils import toxic_elements, toxic_groups

# Grab the tox21 data
tox21 = DFStore().get("/datasets/chem_info/tox21")

# Add molecules to the dataframe
tox21["molecule"] = tox21["smiles"].apply(lambda smiles: Chem.MolFromSmiles(smiles))

# See which molecules are tagged as toxic but not toxic (based on the tox21 dataset)
too_broad = tox21[(tox21["toxic_tag"] == 1) & (tox21["toxic_any"] == 0)].copy()

# Let's check to see if the too_broad molecules still get flagged as toxic
too_broad["toxic_elements"] = too_broad["molecule"].apply(toxic_elements)
too_broad["toxic_groups"] = too_broad["molecule"].apply(toxic_groups)


# Define toxic SMARTS patterns
smarts_patterns = [
    "[N+](=O)[O-]",  # Nitro group (corrected)
    "[C](Cl)(Cl)(Cl)",  # Trichloromethyl group
    "C(=S)N",  # Dithiocarbamate group
    "[Cr](=O)(=O)=O",  # Chromium(VI)
    "[Se][Se]",  # Diselenide group
    "S(=O)(=O)O",  # Sulfonates
    "[N+](C)(C)CCOc1ccccc1",  # Quaternary ammonium with aromatic ether
    "[N+](C)(C)(C)",  # Quaternary ammonium
    "[N+](C)(C)C",  # Tertiary ammonium
    "[n+](N)no",  # Nitro heterocycle with amine
    "[CBr]",  # Organobromides with high halogenation
    "[n+]1cccc(CO)c1",  # Nitroso compound with CO group
]

# Apply Chem.MolFromSmarts() to each pattern
toxic_smarts = [Chem.MolFromSmarts(smarts) for smarts in smarts_patterns]


# Function to apply SMARTS filtering
def filter_known_toxic(smiles, toxic_smarts):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False  # Skip invalid SMILES
    return any(mol.HasSubstructMatch(smarts) for smarts in toxic_smarts)


# Apply filter
# print(f"False Positives: {len(too_broad)}")
# too_broad = too_broad[~too_broad["smiles"].apply(lambda x: filter_known_toxic(x, toxic_smarts))]
# print(f"False Positives after filtering: {len(too_broad)}")

# Print out the results
print(too_broad[["toxic_any", "toxic_elements", "toxic_groups"]].value_counts(dropna=False))

# Grab the ones marked with toxic elements
my_toxic_elements = too_broad[too_broad["toxic_elements"]].copy()

# Print out the results
print(f"False Positives Elements: {len(my_toxic_elements)}")
print(my_toxic_elements["smiles"].values[:10].tolist())

# Grab the ones marked with toxic groups
my_toxic_groups = too_broad[too_broad["toxic_groups"]].copy()

# Print out the results
print(f"False Positives Groups: {len(my_toxic_groups)}")
print(my_toxic_groups["smiles"].values[:10].tolist())

# Too narrow: See which molecules are not tagged as toxic but ARE toxic (based on the tox21 dataset)
too_narrow = tox21[(tox21["toxic_tag"] == 0) & (tox21["toxic_any"] == 1)].copy()

# Let's check to see if the too_narrow molecules are still flagged as not toxic
too_narrow["toxic_elements"] = too_narrow["molecule"].apply(toxic_elements)
too_narrow["toxic_groups"] = too_narrow["molecule"].apply(toxic_groups)


# Print out the results
print(too_narrow[["toxic_any", "toxic_elements", "toxic_groups"]].value_counts(dropna=False))

# Grab the ones not flagged by either toxic elements or toxic groups (so two false negatives)
my_toxic_elements = too_narrow[~too_narrow["toxic_elements"] & ~too_narrow["toxic_groups"]].copy()

# Print out the results
print(f"False Negatives: {len(my_toxic_elements)}")
print(list(my_toxic_elements["smiles"].values[:10].tolist()))
