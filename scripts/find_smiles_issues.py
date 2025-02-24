"""Part of the ML Pipelines will occasionally have an error or issue with particular SMILES strings"""

"""Last Results:

    - Tauomerize: All SMILES strings were valid
    - SMILES to MD: Failed on 1 SMILES string
        - id=B-376 CN1C=CC=C/C1=C\[NH+]=O.[I-]
"""

# Workbench imports
from workbench.api import FeatureSet, Endpoint

feature_set_name = "aqsol_features"
id_column = "id"

# Pull in a dataframe of smiles
fs = FeatureSet(feature_set_name)
df = fs.pull_dataframe()
total_df = df[[id_column, "smiles"]]

end_1 = Endpoint("tautomerize-v0-rt")
print(f"Endpoint: {end_1.uuid}, Instance: {end_1.instance_type}")
end_2 = Endpoint("smiles-to-md-v0-rt")
print(f"Endpoint: {end_2.uuid}, Instance: {end_2.instance_type}")

# Run all the smiles through normal inference (will mark the bad ones)
df = end_1.inference(df)
df = end_2.inference(df)

# Pull out the bad smiles
print("Bad SMILES:")
