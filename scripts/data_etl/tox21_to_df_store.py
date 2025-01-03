import pandas as pd
from rdkit.Chem import PandasTools

# Workbench Imports
from workbench.api.df_store import DFStore
from workbench.utils.chem_utils import compute_morgan_fingerprints, project_fingerprints

# Load SDF file
sdf_file = "/Users/briford/data/workbench/tox21/training/tox21_10k_data_all.sdf"

# Load SDF file directly into a DataFrame
df = PandasTools.LoadSDF(sdf_file, smilesName="smiles", molColName="molecule", includeFingerprints=False)
print(df.head())
print(f"Loaded {len(df)} compounds from {sdf_file}")
print(f"Columns: {df.columns}")
df.rename(columns={"ID": "id"}, inplace=True)

# Column Information
"""
| `Formula`         | Chemical formula of the compound.                                                             |
| `FW`              | Molecular weight (Formula Weight) of the compound.                                            |
| `DSSTox_CID`      | Unique identifier from the Distributed Structure-Searchable Toxicity (DSSTox) database.       |                                               |
| `ID`              | Internal identifier for the compound.                                                         |
| `smiles`          | Simplified Molecular Input Line Entry System representation of the compound's structure.      |
| `molecule`        | Molecular structure information (format may vary). 
| `NR-AR`           | Nuclear receptor assay for androgen receptor.     
| `SR-HSE`          | Stress response assay for heat shock element.  |
| `SR-ARE`          | Stress response assay for antioxidant response element.                                       |
| `NR-Aromatase`    | Nuclear receptor assay for aromatase enzyme.                                                  |
| `NR-ER-LBD`       | Nuclear receptor assay for estrogen receptor ligand-binding domain.                           |
| `NR-AhR`          | Nuclear receptor assay for aryl hydrocarbon receptor.                                         |
| `SR-MMP`          | Stress response assay for mitochondrial membrane potential.                                   |
| `NR-ER`           | Nuclear receptor assay for estrogen receptor.                                                 |
| `NR-PPAR-gamma`   | Nuclear receptor assay for peroxisome proliferator-activated receptor gamma.                  |
| `SR-p53`          | Stress response assay for p53 tumor suppressor protein.                                       |
| `SR-ATAD5`        | Stress response assay for ATAD5 (ATPase family AAA domain-containing protein 5).              |
| `NR-AR-LBD`       | Nuclear receptor assay for androgen receptor ligand-binding domain.                           |
"""

# Convert Formula weight to numeric
df["FW"] = pd.to_numeric(df["FW"], errors="coerce")

# Full list of assay columns for toxicity detection (0/1/NaN) for values
assay_cols = [
    "NR-AR",  # Nuclear Receptor - Androgen Receptor
    "SR-HSE",  # Stress Response - Heat Shock Element
    "SR-ARE",  # Stress Response - Antioxidant Response Element
    "NR-Aromatase",  # Nuclear Receptor - Aromatase
    "NR-ER-LBD",  # Nuclear Receptor - Estrogen Receptor (Ligand Binding Domain)
    "NR-AhR",  # Nuclear Receptor - Aryl Hydrocarbon Receptor
    "SR-MMP",  # Stress Response - Mitochondrial Membrane Potential
    "NR-ER",  # Nuclear Receptor - Estrogen Receptor
    "NR-PPAR-gamma",  # Nuclear Receptor - Peroxisome Proliferator-Activated Receptor Gamma
    "SR-p53",  # Stress Response - p53 Tumor Suppressor Protein
    "SR-ATAD5",  # Stress Response - ATPase Family AAA Domain-Containing Protein 5
    "NR-AR-LBD",  # Nuclear Receptor - Androgen Receptor (Ligand Binding Domain)
]

# Convert to numeric, coercing errors to NaN
df[assay_cols] = df[assay_cols].apply(pd.to_numeric, errors="coerce")

# Set is_toxic to 1 if any of the toxicity columns has a 1
df["toxic_any"] = (df[assay_cols] == 1).any(axis=1).astype(int)
print(df["toxic_any"].value_counts(dropna=False))

# Do both TSNE and UMAP projections
df = compute_morgan_fingerprints(df, radius=2)
df = project_fingerprints(df, projection="TSNE")
df.rename(columns={"x": "x_tsne", "y": "y_tsne"}, inplace=True)
df = project_fingerprints(df, projection="UMAP")

# Convert compound tags to string and check for substrings, then convert to 0/1
df["toxic_tag"] = df["tags"].astype(str).str.contains("toxic").astype(int)
df["druglike_tag"] = df["tags"].astype(str).str.contains("druglike").astype(int)

# Now store the DataFrame in the DFStore
df_store = DFStore()
df.drop(columns=["molecule"], inplace=True)  # We can't serialize the molecule column
df_store.upsert("/datasets/chem_info/tox21_training", df)
