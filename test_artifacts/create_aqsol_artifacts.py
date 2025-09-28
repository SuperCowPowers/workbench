"""This Script creates the AQSol (Public) Artifacts in AWS/Workbench

DataSources:
    - aqsol_data

FeatureSets:
    - aqsol_features
    - aqsol_mol_descriptors

Models:
    - aqsol-regression
    - aqsol-class
    - aqsol-mol-regression
    - aqsol-mol-class
    - smiles-to-taut-md-stereo-v0
    - smiles-to-fingerprints-v0
    - tautomerize-v0

Endpoints:
    - aqsol-regression
    - aqsol-class
    - aqsol-mol-regression
    - aqsol-mol-class
    - smiles-to-taut-md-stereo-v0
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

    aqsol_features = [
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
    # Create the aqsol solubility regression Model
    if recreate or not Model("aqsol-regression").exists():
        # Compute our features
        feature_set = FeatureSet("aqsol_features")
        m = feature_set.to_model(
            name="aqsol-regression",
            model_type=ModelType.REGRESSOR,
            target_column="solubility",
            feature_list=aqsol_features,
            description="AQSol Regression Model",
            tags=["aqsol", "regression"],
        )
        m.set_owner("test")

    # Create the aqsol regression Endpoint
    if recreate or not Endpoint("aqsol-regression").exists():
        m = Model("aqsol-regression")
        end = m.to_endpoint(tags=["aqsol", "regression"])
        end.set_owner("test")
        end.auto_inference(capture=True)

    # Create the aqsol solubility classification Model
    if recreate or not Model("aqsol-class").exists():
        # Compute our features
        feature_set = FeatureSet("aqsol_features")
        m = feature_set.to_model(
            name="aqsol-class",
            model_type=ModelType.CLASSIFIER,
            target_column="solubility_class",
            feature_list=aqsol_features,
            description="AQSol Classification Model",
            tags=["aqsol", "classification"],
        )
        m.set_owner("test")
        m.set_class_labels(["low", "medium", "high"])

    # Create the aqsol classification Endpoint
    if recreate or not Endpoint("aqsol-class").exists():
        m = Model("aqsol-class")
        end = m.to_endpoint(tags=["aqsol", "classification"])
        end.set_owner("test")
        end.auto_inference(capture=True)

    #
    # Molecular Descriptor Artifacts
    #
    # Create the rdkit FeatureSet (this is an example of using lower level classes)
    if recreate or not FeatureSet("aqsol_mol_descriptors").exists():
        df = DataSource("aqsol_data").pull_dataframe()
        end = Endpoint("smiles-to-taut-md-stereo-v0")
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
            "has_stereo",
        ]
        feature_list = [f for f in feature_set.columns if f not in exclude]
        m = feature_set.to_model(
            name="aqsol-mol-class",
            model_type=ModelType.CLASSIFIER,
            target_column="solubility_class",
            feature_list=feature_list,
            description="AQSol Descriptor Classification Model",
            tags=["aqsol", "classification"],
        )
        m.set_class_labels(["low", "medium", "high"])

    # Create the Molecular Descriptor Regression Endpoint
    if recreate or not Endpoint("aqsol-mol-regression").exists():
        m = Model("aqsol-mol-regression")
        end = m.to_endpoint(tags=["aqsol", "mol", "regression"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # Create the Molecular Descriptor Classification Endpoint
    if recreate or not Endpoint("aqsol-mol-class").exists():
        m = Model("aqsol-mol-class")
        end = m.to_endpoint(tags=["aqsol", "mol", "classification"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # A 'Model' to Compute Molecular Descriptors Features
    if recreate or not Model("smiles-to-taut-md-stereo-v0").exists():
        script_path = get_custom_script_path("chem_info", "molecular_descriptors.py")
        feature_set = FeatureSet("aqsol_features")
        feature_set.to_model(
            name="smiles-to-taut-md-stereo-v0",
            model_type=ModelType.TRANSFORMER,
            feature_list=["smiles"],
            description="Smiles to Molecular Descriptors",
            tags=["smiles", "molecular descriptors", "stereo"],
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

    # Endpoints for our Transformer/Custom Models
    if recreate or not Endpoint("smiles-to-taut-md-stereo-v0").exists():
        m = Model("smiles-to-taut-md-stereo-v0")
        end = m.to_endpoint(tags=["smiles", "molecular descriptors", "stereo"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    if recreate or not Endpoint("smiles-to-fingerprints-v0").exists():
        m = Model("smiles-to-fingerprints-v0")
        end = m.to_endpoint(tags=["smiles", "morgan fingerprints"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # Fingerprint FeatureSet
    if recreate or not FeatureSet("aqsol_fingerprints").exists():

        # Run the smiles through the fingerprint feature endpoint
        end = Endpoint("smiles-to-fingerprints-v0")
        feature_set = FeatureSet("aqsol_features")
        input_df = feature_set.pull_dataframe()
        fingerprint_df = end.inference(input_df)
        to_features = PandasToFeatures("aqsol_fingerprints")
        to_features.set_output_tags(["aqsol", "public", "fingerprints"])
        to_features.set_input(fingerprint_df, id_column="id")
        to_features.transform()

        # Set our compressed features
        feature_set = FeatureSet("aqsol_fingerprints")
        feature_set.set_compressed_features(["fingerprint"])

    # Fingerprint Model
    if recreate or not Model("aqsol-fingerprints").exists():

        # Create the Fingerprint Model
        feature_set = FeatureSet("aqsol_fingerprints")
        feature_set.to_model(
            name="aqsol-fingerprints",
            model_type=ModelType.REGRESSOR,
            feature_list=["fingerprint"],
            target_column="solubility",
            description="Morgan Fingerprints Model",
            tags=["smiles", "fingerprints"],
            train_all_data=True,
        )

    # Fingerprint Endpoint
    if recreate or not Endpoint("aqsol-fingerprints").exists():
        m = Model("aqsol-fingerprints")
        end = m.to_endpoint(tags=["smiles", "fingerprints"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # Fingerprint + Other Features Model
    if recreate or not Model("aqsol-fingerprints-plus").exists():

        # Grab the features from an existing model
        features = Model("aqsol-regression").features() + ["fingerprint"]

        # Create the Fingerprint Model
        feature_set = FeatureSet("aqsol_fingerprints")
        model = feature_set.to_model(
            name="aqsol-fingerprints-plus",
            model_type=ModelType.REGRESSOR,
            target_column="solubility",
            feature_list=features,
            description="Morgan Fingerprints + Features Model",
            tags=["smiles", "fingerprints", "plus"],
            train_all_data=True,
        )

    # Fingerprint + Other Features Endpoint
    if recreate or not Endpoint("aqsol-fingerprints-plus").exists():
        m = Model("aqsol-fingerprints-plus")
        end = m.to_endpoint(tags=["smiles", "fingerprints", "plus"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)

    # Fingerprint + Other Features CLASSIFICATION Model
    if recreate or not Model("aqsol-fingerprints-plus-class").exists():

        # Grab the features from an existing model
        features = Model("aqsol-regression").features() + ["fingerprint"]

        # Create the Fingerprint Model
        feature_set = FeatureSet("aqsol_fingerprints")
        model = feature_set.to_model(
            name="aqsol-fingerprints-plus-class",
            model_type=ModelType.CLASSIFIER,
            target_column="solubility_class",
            feature_list=features,
            description="Morgan Fingerprints + Features Classification Model",
            tags=["smiles", "fingerprints", "plus"],
            train_all_data=True,
        )
        model.set_class_labels(["low", "medium", "high"])

    # Fingerprint + Other Features CLASSIFICATION Endpoint
    if recreate or not Endpoint("aqsol-fingerprints-plus-class").exists():
        m = Model("aqsol-fingerprints-plus-class")
        end = m.to_endpoint(tags=["smiles", "fingerprints", "plus"])

        # Run inference on the endpoint
        end.auto_inference(capture=True)
