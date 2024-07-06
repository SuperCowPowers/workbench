"""Create Solubility Models in AWS/SageWorks

We're using Pipelines to create different set of ML Artifacts in SageWorks
"""

import pandas as pd
import numpy as np
import logging

from sageworks.api.data_source import DataSource
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType
from sageworks.api.endpoint import Endpoint

from sageworks.core.transforms.data_to_features.light.molecular_descriptors import (
    MolecularDescriptors,
)
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker
from sageworks.api.pipeline import Pipeline
from sageworks.utils.pandas_utils import stratified_split

log = logging.getLogger("sageworks")

# Set our pipeline
pipeline_name = "test_solubility_class_nightly_100_v0"


if __name__ == "__main__":
    # This forces a refresh on all the data we get from the AWs Broker
    AWSServiceBroker().get_all_metadata(force_refresh=True)

    # Grab all the information from the Pipeline (as a dictionary)
    pipe = Pipeline(pipeline_name).pipeline

    # Get all the pipeline information
    s3_path = pipe["data_source"]["input"]
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
    pipeline_id = pipe["pipeline"]

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the aqsol_data DataSource
    if recreate or not DataSource(data_source_name).exists():
        # Grab the input and add some columns
        df = DataSource(data_source_input).pull_dataframe()

        # Remove 'weird' values
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

        # Now we'll create the DataSource with the new column
        DataSource(df, name=data_source_name, tags=data_source_tags)

    #
    # Molecular Descriptor Artifacts
    #
    # Create the rdkit FeatureSet (this is an example of using lower level classes)
    if recreate or not FeatureSet(feature_set_name).exists():

        rdkit_features = MolecularDescriptors(data_source_name, feature_set_name)
        rdkit_features.set_output_tags(feature_set_tags)
        rdkit_features.transform(id_column="udm_mol_id")

    # Set the holdout ids for the FeatureSet
    fs = FeatureSet(feature_set_name)

    # Hold out logic (might be a list of ids or a stratified split)
    if isinstance(holdout, list):
        fs.set_holdout_ids("udm_mol_id", holdout)
    else:
        # Stratified Split, so we need to pull the parameters from the string
        test_size = float(holdout.split(":")[1])
        column_name = holdout.split(":")[2]
        df = fs.pull_dataframe()[["udm_mol_id", column_name]]

        # Perform the stratified split and set the hold out ids
        train, test = stratified_split(df, column_name=column_name, test_size=test_size)
        fs.set_holdout_ids("udm_mol_id", test["udm_mol_id"].tolist())

    # Create the Model
    model_type = ModelType(model_type_str)
    if recreate or not Model(model_name).exists():
        feature_set = FeatureSet(feature_set_name)
        feature_set.to_model(
            model_type,
            target_column=model_target,
            name=model_name,
            feature_list=model_features,
            tags=model_tags,
        )

    # Create the Endpoint
    if recreate or not Endpoint(endpoint_name).exists():
        m = Model(model_name)
        m.to_endpoint(name=endpoint_name, tags=endpoint_tags)
        end = Endpoint(endpoint_name)
        end.auto_inference(capture=True)
