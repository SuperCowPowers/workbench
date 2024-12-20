"""Test Script for the FeatureResolution Class"""

import pandas as pd
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model
from workbench.algorithms.dataframe.feature_resolution import FeatureResolution

# Set some pandas options
pd.options.display.max_columns = None
pd.options.display.width = 1000

# Grab a test dataframe
fs = FeatureSet("aqsol_mol_descriptors")
test_df = fs.pull_dataframe()

# Get the Model (for the target and feature columns)
model = Model("aqsol-mol-regression")
target = model.target()
features = model.features()

# Create the class and run the report
resolution = FeatureResolution(test_df, features=features, target_column=target, id_column=fs.id_column)

# Add some output columns
output_columns = ["solubility", "solubility_class", "smiles"]
output_df = resolution.compute(within_distance=0.01, min_target_difference=1.0, output_columns=output_columns)

# Print the output
print(output_df.head())
