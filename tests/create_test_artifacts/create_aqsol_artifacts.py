"""This Script creates the AQSol (Public) Artifacts in AWS/Workbench

DataSources:
    - aqsol_data

FeatureSets:
    - aqsol_features
    - aqsol_mol_descriptors

Models:
    - aqsol-regression
    - aqsol-mol-regression
    - aqsol-mol-class
    - smiles-to-md-v0
    - smiles-to-fingerprints-v0
    - tautomerize-v0

Endpoints:
    - aqsol-regression
    - aqsol-mol-regression
    - aqsol-mol-class
    - smiles-to-md-v0
    - smiles-to-fingerprints-v0
    - tautomerize-v0
"""

import logging
import pandas as pd
import awswrangler as wr

from workbench.api import DataSource, FeatureSet, Model, ModelType, Endpoint
from workbench.core.transforms.pandas_transforms import PandasToFeatures
from workbench.utils.model_utils import get_custom_script_path

log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Get the path to the dataset in S3
    s3_path = "s3://workbench-public-data/comp_chem/aqsol_public_data.csv"

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
            name="aqsol-regression",
            model_type=ModelType.REGRESSOR,
            target_column="solubility",
            feature_list=feature_list,
            description="AQSol Regression Model",
            tags=["aqsol", "regression"],
        )

    # Create the aqsol regression Endpoint
    if recreate or not Endpoint("aqsol-regression").exists():
        m = Model("aqsol-regression")
        m.to_endpoint(name="aqsol-regression", tags=["aqsol", "regression"])

    #
    # Molecular Descriptor Artifacts
    #
    # Create the rdkit FeatureSet (this is an example of using lower level classes)
    if recreate or not FeatureSet("aqsol_mol_descriptors").exists():
        df = DataSource("aqsol_data").pull_dataframe()
        end = Endpoint("smiles-to-md-v0")
        mol_df = end.inference(df)
        to_features = PandasToFeatures("aqsol_mol_descriptors")
        to_features.set_output_tags(["aqsol", "public"])
        to_features.set_input(mol_df, id_column="id")
        to_features.transform()

    # Create the Molecular Descriptor based Regression Model
    if recreate or not Model("aqsol-mol-regression").exists():
        # Compute our features
        feature_set = FeatureSet("aqsol_mol_descriptors")
        exclude = [
            "id",
            "name",
            "inchi",
            "inchikey",
            "smiles",
            "group",
            "solubility",
            "solubility_class",
            "rotratio",  # rotratio is often string type
            "ocurrences",
            "sd",
        ]
        feature_list = [f for f in feature_set.columns if f not in exclude]
        feature_set.to_model(
            name="aqsol-mol-regression",
            model_type=ModelType.REGRESSOR,
            target_column="solubility",
            feature_list=feature_list,
            description="AQSol Descriptor Regression Model",
            tags=["aqsol", "regression"],
        )

    # Create the Molecular Descriptor based Classification Model
    if recreate or not Model("aqsol-mol-class").exists():
        # Compute our features
        feature_set = FeatureSet("aqsol_mol_descriptors")
        exclude = [
            "id",
            "name",
            "inchi",
            "inchikey",
            "smiles",
            "group",
            "solubility",
            "solubility_class",
            "rotratio",  # rotratio is often string type
            "ocurrences",
            "sd",
        ]
        feature_list = [f for f in feature_set.columns if f not in exclude]
        feature_set.to_model(
            name="aqsol-mol-class",
            model_type=ModelType.CLASSIFIER,
            target_column="solubility_class",
            feature_list=feature_list,
            description="AQSol Descriptor Classification Model",
            tags=["aqsol", "classification"],
        )

    # Create the Molecular Descriptor Regression Endpoint
    if recreate or not Endpoint("aqsol-mol-regression").exists():
        m = Model("aqsol-mol-regression")
        end = m.to_endpoint(name="aqsol-mol-regression", tags=["aqsol", "mol", "regression"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # Create the Molecular Descriptor Classification Endpoint
    if recreate or not Endpoint("aqsol-mol-class").exists():
        m = Model("aqsol-mol-class")
        end = m.to_endpoint(name="aqsol-mol-class", tags=["aqsol", "mol", "classification"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # A 'Model' to Compute Molecular Descriptors Features
    if recreate or not Model("smiles-to-md-v0").exists():
        script_path = get_custom_script_path("chem_info", "molecular_descriptors.py")
        feature_set = FeatureSet("aqsol_features")
        feature_set.to_model(
            name="smiles-to-md-v0",
            model_type=ModelType.TRANSFORMER,
            feature_list=["smiles"],
            description="Smiles to Molecular Descriptors",
            tags=["smiles", "molecular descriptors"],
            custom_script=script_path,
        )

    # A 'Model' to Compute Morgan Fingerprints Features
    if recreate or not Model("smiles-to-fingerprints-v0").exists():
        script_path = get_custom_script_path("chem_info", "morgan_fingerprints.py")
        feature_set = FeatureSet("aqsol_features")
        feature_set.to_model(
            name="smiles-to-fingerprints-v0",
            model_type=ModelType.TRANSFORMER,
            feature_list=["smiles"],
            description="Smiles to Morgan Fingerprints",
            tags=["smiles", "morgan fingerprints"],
            custom_script=script_path,
        )

    # A 'Model' to Tautomerize Smiles
    if recreate or not Model("tautomerize-v0").exists():
        script_path = get_custom_script_path("chem_info", "tautomerize.py")
        feature_set = FeatureSet("aqsol_features")
        feature_set.to_model(
            name="tautomerize-v0",
            model_type=ModelType.TRANSFORMER,
            feature_list=["smiles"],
            description="Tautomerize Smiles",
            tags=["smiles", "tautomerization"],
            custom_script=script_path,
        )

    # Endpoints for our Transformer/Custom Models
    if recreate or not Endpoint("smiles-to-md-v0").exists():
        m = Model("smiles-to-md-v0")
        end = m.to_endpoint(name="smiles-to-md-v0", tags=["smiles", "molecular descriptors"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    if recreate or not Endpoint("smiles-to-fingerprints-v0").exists():
        m = Model("smiles-to-fingerprints-v0")
        end = m.to_endpoint(name="smiles-to-fingerprints-v0", tags=["smiles", "morgan fingerprints"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    if recreate or not Endpoint("tautomerize-v0").exists():
        m = Model("tautomerize-v0")
        end = m.to_endpoint(name="tautomerize-v0", tags=["smiles", "tautomerization"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)
