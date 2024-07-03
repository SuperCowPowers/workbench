"""Example Script for the FeatureResolution Class"""

import numpy as np
import pandas as pd

from sageworks.api.feature_set import FeatureSet
from sageworks.algorithms.dataframe.feature_resolution import FeatureResolution

# Grab a test dataframe
fs = FeatureSet("solubility_featurized_ds")
test_df = fs.pull_dataframe()

# Convert to logS
# Note: This will make 0 -> -16
test_df["udm_asy_res_value"] = test_df["udm_asy_res_value"].replace(0, 1e-10)
test_df["log_s"] = np.log10(test_df["udm_asy_res_value"] / 1e6)

target_column = "log_s"
meta = [
    "write_time",
    "api_invocation_time",
    "is_deleted",
    "udm_asy_protocol",
    "udm_asy_cnd_format",
    "std_dev",
    "count",
    "udm_mol_id",
    "udm_asy_date",
    "udm_prj_code",
    "udm_asy_cnd_target",
    "udm_asy_cnd_time_hr",
    "smiles",
    "udm_mol_bat_slt_smiles",
    "udm_mol_bat_slv_smiles",
    "operator",
    "class",
    "event_time",
]
exclude = ["log_s", "udm_asy_res_value", "udm_mol_bat_id"] + meta
feature_columns = [c for c in test_df.columns if c not in exclude]
print(f"Num Features {len(feature_columns)}")

output_columns = [
    "smiles",
    "udm_asy_protocol",
    "udm_asy_cnd_target",
    "udm_asy_cnd_format",
    "udm_asy_date",
    "udm_prj_code",
    "udm_mol_bat_slt_ratio",
    "udm_mol_bat_slt_smiles",
    "udm_mol_bat_slv_ratio",
    "udm_mol_bat_slv_smiles",
    "udm_asy_cnd_time_hr",
]

# Create the class and run the report
resolution = FeatureResolution(
    test_df,
    features=feature_columns,
    target_column=target_column,
    id_column="udm_mol_bat_id",
)
output_df = resolution.compute(within_distance=0.01, min_target_difference=2.0, output_columns=output_columns)

# Print the output
pd.options.display.max_columns = None
pd.options.display.width = 1000
print(output_df.head())
