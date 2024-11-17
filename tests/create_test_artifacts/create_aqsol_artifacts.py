"""This Script creates the AQSol (Public) Artifacts in AWS/SageWorks

DataSources:
    - aqsol_data
FeatureSets:
    - aqsol_features
Models:
    - aqsol-regression
Endpoints:
    - aqsol-regression-end
"""

import logging
import pandas as pd
import awswrangler as wr

from sageworks.api.data_source import DataSource
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType
from sageworks.api.endpoint import Endpoint

from sageworks.core.transforms.data_to_features.light.molecular_descriptors import MolecularDescriptors

log = logging.getLogger("sageworks")


if __name__ == "__main__":

    # Get the path to the dataset in S3
    s3_path = "s3://sageworks-public-data/comp_chem/aqsol_public_data.csv"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the aqsol_data DataSource
    if recreate or not DataSource("aqsol_data").exists():
        # We could create a Datasource directly,  but we're going to add a column to the data
        df = wr.s3.read_csv(s3_path)

        # Create a solubility classification column
        bins = [-float("inf"), -5, -4, float("inf")]
        labels = ["low", "medium", "high"]
        df["solubility_class"] = pd.cut(df["Solubility"], bins=bins, labels=labels)

        # Now we'll create the DataSource with the new column
        DataSource(df, name="aqsol_data")

    # Create the aqsol_features FeatureSet
    if recreate or not FeatureSet("aqsol_features").exists():
        ds = DataSource("aqsol_data")
        ds.to_features("aqsol_features", id_column="id", tags=["aqsol", "public"])

    # Create the aqsol solubility regression Model
    if recreate or not Model("aqsol-regression").exists():
        # Compute our features
        feature_set = FeatureSet("aqsol_features")
        feature_list = [
            "molwt",
            "mollogp",
            "molmr",
            "heavyatomcount",
            "numhacceptors",
            "numhdonors",
            "numheteroatoms",
            "numrotatablebonds",
            "numvalenceelectrons",
            "numaromaticrings",
            "numsaturatedrings",
            "numaliphaticrings",
            "ringcount",
            "tpsa",
            "labuteasa",
            "balabanj",
            "bertzct",
        ]
        feature_set.to_model(
            ModelType.REGRESSOR,
            target_column="solubility",
            name="aqsol-regression",
            feature_list=feature_list,
            description="AQSol Regression Model",
            tags=["aqsol", "regression"],
        )

    # Create the aqsol regression Endpoint
    if recreate or not Endpoint("aqsol-regression-end").exists():
        m = Model("aqsol-regression")
        m.to_endpoint(name="aqsol-regression-end", tags=["aqsol", "regression"])

    #
    # Molecular Descriptor Artifacts
    #
    # Create the rdkit FeatureSet (this is an example of using lower level classes)
    if recreate or not FeatureSet("aqsol_mol_descriptors").exists():
        rdkit_features = MolecularDescriptors("aqsol_data", "aqsol_mol_descriptors")
        rdkit_features.set_output_tags(["aqsol", "public"])
        query = "SELECT id, solubility, solubility_class, smiles FROM aqsol_data"
        rdkit_features.transform(target_column="solubility", id_column="id", query=query)

    # Create the Molecular Descriptor based Regression Model
    if recreate or not Model("aqsol-mol-regression").exists():
        # Compute our features
        feature_set = FeatureSet("aqsol_mol_descriptors")
        exclude = ["id", "smiles", "solubility", "solubility_class"]
        feature_list = [f for f in feature_set.columns if f not in exclude]
        feature_set.to_model(
            ModelType.REGRESSOR,
            target_column="solubility",
            name="aqsol-mol-regression",
            feature_list=feature_list,
            description="AQSol Descriptor Regression Model",
            tags=["aqsol", "regression"],
        )

    # Create the Molecular Descriptor based Classification Model
    if recreate or not Model("aqsol-mol-class").exists():
        # Compute our features
        feature_set = FeatureSet("aqsol_mol_descriptors")
        exclude = ["id", "smiles", "solubility", "solubility_class"]
        feature_list = [f for f in feature_set.columns if f not in exclude]
        feature_set.to_model(
            ModelType.CLASSIFIER,
            target_column="solubility_class",
            name="aqsol-mol-class",
            feature_list=feature_list,
            description="AQSol Descriptor Classification Model",
            tags=["aqsol", "classification"],
        )

    # Create the Molecular Descriptor Regression Endpoint
    if recreate or not Endpoint("aqsol-mol-regression-end").exists():
        m = Model("aqsol-mol-regression")
        end = m.to_endpoint(name="aqsol-mol-regression-end", tags=["aqsol", "mol", "regression"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # Create the Molecular Descriptor Classification Endpoint
    if recreate or not Endpoint("aqsol-mol-class-end").exists():
        m = Model("aqsol-mol-class")
        end = m.to_endpoint(name="aqsol-mol-class-end", tags=["aqsol", "mol", "classification"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)
