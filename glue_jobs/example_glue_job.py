# Example Glue Job that goes from CSV to Model/Endpoint
import sys
import numpy as np
import pandas as pd

# SageWorks Imports
from sageworks.api.data_source import DataSource
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType
from sageworks.api.endpoint import Endpoint
from sageworks.core.transforms.data_to_features.light.molecular_descriptors import (
    MolecularDescriptors,
)
from sageworks.core.transforms.pandas_transforms.pandas_to_features import (
    PandasToFeatures,
)
from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.glue_utils import get_resolved_options

# Convert Glue Job Args to a Dictionary
glue_args = get_resolved_options(sys.argv)

# Set the SAGEWORKS_BUCKET for the ConfigManager
cm = ConfigManager()
cm.set_config("SAGEWORKS_BUCKET", glue_args["sageworks-bucket"])
cm.set_config("REDIS_HOST", glue_args["redis-host"])

# Create a new Data Source from an S3 Path
# source_path = "s3://idb-forest-sandbox/physchemproperty/LogS/Null/gen_processed/2024_03_07_id_smiles.csv"
source_path = (
    "s3://idb-forest-sandbox/physchemproperty/assay_processed_collection/solubility/all/2024_03_07_id_smiles.csv"
)
my_data = DataSource(source_path, name="solubility_test_data")

# Pull the dataframe from the Data Source
df = DataSource("solubility_test_data").pull_dataframe()

# Convert to logS
# Note: This will make 0 -> -16
df["udm_asy_res_value"] = df["udm_asy_res_value"].replace(0, 1e-10)
df["log_s"] = np.log10(df["udm_asy_res_value"] / 1e6)
df["log_s"] = df["udm_asy_res_value"]

# Create a solubility classification column
bins = [-float("inf"), -5, -4, float("inf")]
labels = ["low", "medium", "high"]
df["sol_class"] = pd.cut(df["log_s"], bins=bins, labels=labels)

# Compute molecular descriptors
molecular_features = MolecularDescriptors("solubility_test_data", "solubility_test_features")

# Okay we're going to use the guts of the class without actually doing the DS to FS transformation
molecular_features.input_df = df[:100]
molecular_features.transform_impl()
output_df = molecular_features.output_df
print(output_df.head())

# Create a Feature Set
to_features = PandasToFeatures("solubility_test_features")
to_features.set_input(output_df, id_column="udm_mol_bat_id")
to_features.set_output_tags(["test", "solubility"])
to_features.transform()


"""
DataSource(source_path, name="solubility_test_data")

# Create a Feature Set
molecular_features = MolecularDescriptors("solubility_test_data", "solubility_test_features")
molecular_features.set_output_tags(["test", "solubility", "molecular_descriptors"])
query = "SELECT udm_mol_bat_id,  udm_asy_protocol, udm_prj_code, udm_asy_res_value, smiles FROM solubility_test_data"
molecular_features.transform(target_column="solubility", id_column="udm_mol_bat_id", query=query)
"""

"""
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
"""
