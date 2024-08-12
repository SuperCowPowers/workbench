"""Create Solubility Models in AWS/SageWorks

We're using Pipelines to create different set of ML Artifacts in SageWorks
"""

import sys
import pandas as pd
import numpy as np
import logging

from sageworks.api import DataSource, FeatureSet, Model, Endpoint
from sageworks.api.model import ModelType
from sageworks.core.transforms.data_to_features.light.molecular_descriptors import MolecularDescriptors
from sageworks.api.pipeline import Pipeline

from sageworks.utils.config_manager import ConfigManager
from sageworks.utils.glue_utils import get_resolved_options

# Convert Glue Job Args to a Dictionary
glue_args = get_resolved_options(sys.argv)

# Set the SageWorks Config (needs to be done early)
cm = ConfigManager()
cm.set_config("SAGEWORKS_BUCKET", glue_args["sageworks-bucket"])
cm.set_config("REDIS_HOST", glue_args["redis-host"])
log = logging.getLogger("sageworks")

# Set our pipeline
pipeline_name = "test_solubility_class_nightly_100_v0"


# A bit of specific processing (maybe put in utils or something)
def solubility_processing(df):
    # Remove 'weird' solubility values
    log.important("Removing 'weird' values from the solubility data")
    log.important(f"Original Shape: {df.shape}")
    df = df[df["udm_asy_res_value"] != 4.7]
    df = df[df["udm_asy_res_value"] != 0]
    log.important(f"New Shape: {df.shape}")

    # Compute the log of the solubility
    df["udm_asy_res_value"] = df["udm_asy_res_value"].replace(0, 1e-10)
    df["log_s"] = np.log10(df["udm_asy_res_value"] / 1e6)

    # Create a solubility classification column
    bins = [-float("inf"), -5, -4, float("inf")]
    labels = ["low", "medium", "high"]
    df["solubility_class"] = pd.cut(df["log_s"], bins=bins, labels=labels)
    return df


if __name__ == "__main__":

    # Grab all the information from the Pipeline (as a dictionary)
    pipe = Pipeline(pipeline_name).pipeline

    # Get all the pipeline information
    id_column = pipe["data_source"]["id_column"]
    model_features = pipe["model"]["feature_list"]
    data_source_input = pipe["data_source"]["input"]
    data_source_name = pipe["data_source"]["name"]
    data_source_tags = pipe["data_source"]["tags"]
    feature_set_name = pipe["feature_set"]["name"]
    feature_set_tags = pipe["feature_set"]["tags"]
    holdout = pipe["feature_set"]["holdout"]
    model_name = pipe["model"]["name"]
    model_type_str = pipe["model"]["model_type"]
    model_tags = pipe["model"]["tags"]
    model_target = pipe["model"]["target_column"]
    endpoint_name = pipe["endpoint"]["name"]
    endpoint_tags = pipe["endpoint"]["tags"]
    pipeline_name = pipe["pipeline"]

    # Grab the data from the input DataSource
    df = DataSource(data_source_input).pull_dataframe()

    # A bit of specific processing
    df = solubility_processing(df)

    # Now we'll create the DataSource with the new column
    DataSource(df, name=data_source_name, tags=data_source_tags)

    # Molecular Descriptor Artifacts
    rdkit_features = MolecularDescriptors(data_source_name, feature_set_name)
    rdkit_features.set_output_tags(feature_set_tags)
    rdkit_features.transform(id_column=id_column)

    # Set the holdout ids for the FeatureSet (not needed for this example)
    fs = FeatureSet(feature_set_name)

    # Create the Model
    model_type = ModelType(model_type_str)
    feature_set = FeatureSet(feature_set_name)
    feature_set.to_model(
        model_type,
        target_column=model_target,
        name=model_name,
        feature_list=model_features,
        tags=model_tags,
    )

    # Create the Endpoint
    m = Model(model_name)
    m.set_pipeline(pipeline_name)
    m.to_endpoint(name=endpoint_name, tags=endpoint_tags)
    end = Endpoint(endpoint_name)
    end.auto_inference(capture=True)
