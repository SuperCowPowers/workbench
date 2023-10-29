"""This Script creates the Classification Artifacts in AWS/SageWorks

DataSources:
    - aqsol_data
FeatureSets:
    - aqsol_features
Models:
    - aqsol-regression
Endpoints:
    - aqsol-regression-end
"""
from sageworks.artifacts.data_sources.data_source import DataSource
from sageworks.artifacts.feature_sets.feature_set import FeatureSet
from sageworks.artifacts.models.model import Model, ModelType
from sageworks.artifacts.endpoints.endpoint import Endpoint

from sageworks.transforms.data_loaders.light.s3_to_data_source_light import S3ToDataSourceLight
from sageworks.transforms.data_to_features.light.data_to_features_light import DataToFeaturesLight
from sageworks.transforms.data_to_features.light.rdkit_descriptors import RDKitDescriptors
from sageworks.transforms.features_to_model.features_to_model import FeaturesToModel
from sageworks.transforms.model_to_endpoint.model_to_endpoint import ModelToEndpoint

if __name__ == "__main__":
    # Get the path to the dataset in S3
    s3_path = "s3://sageworks-public-data/comp_chem/aqsol_public_data.csv"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the aqsol_data DataSource
    if recreate or not DataSource("aqsol_data").exists():
        to_data_source = S3ToDataSourceLight(s3_path, "aqsol_data")
        to_data_source.set_output_tags(["aqsol", "public"])
        to_data_source.transform()

    # Create the aqsol_features FeatureSet
    if recreate or not FeatureSet("aqsol_features").exists():
        data_to_features = DataToFeaturesLight("aqsol_data", "aqsol_features")
        data_to_features.set_output_tags(["aqsol", "public"])
        data_to_features.transform(id_column="id", target_column="solubility")

    # Create the aqsol solubility regression Model
    if recreate or not Model("aqsol-regression").exists():
        # Compute our features
        feature_set = FeatureSet("aqsol_features")
        feature_list = [
            "sd",
            "ocurrences",
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
        # exclude = ["id", "name", "inchi", "inchikey", "smiles", "solubility", "group"]
        # feature_list = [f for f in feature_set.column_names() if f not in exclude]
        features_to_model = FeaturesToModel("aqsol_features", "aqsol-regression", model_type=ModelType.REGRESSOR)
        features_to_model.set_output_tags(["aqsol", "regression"])
        features_to_model.transform(
            target_column="solubility", description="AQSol Regression Model", feature_list=feature_list
        )

    # Create the aqsol regression Endpoint
    if recreate or not Endpoint("aqsol-regression-end").exists():
        model_to_endpoint = ModelToEndpoint("aqsol-regression", "aqsol-regression-end", serverless=False)
        model_to_endpoint.set_output_tags(["aqsol", "regression"])
        model_to_endpoint.transform()

    #
    # RDKIT Descriptor Artifacts
    #
    # Create the rdkit FeatureSet
    if recreate or not FeatureSet("aqsol_rdkit_features").exists():
        rdkit_features = RDKitDescriptors("aqsol_data", "aqsol_rdkit_features")
        rdkit_features.set_output_tags(["aqsol", "public", "rdkit"])
        query = "SELECT id, solubility, smiles FROM aqsol_data"
        rdkit_features.transform(target_column="solubility", id_column="udm_mol_bat_id", query=query)

    # Create the RDKIT based  regression Model
    if recreate or not Model("aqsol-rdkit-regression").exists():
        # Compute our features
        feature_set = FeatureSet("aqsol_rdkit_features")
        exclude = ["id", "smiles", "solubility"]
        feature_list = [f for f in feature_set.column_names() if f not in exclude]
        features_to_model = FeaturesToModel(
            "aqsol_rdkit_features", "aqsol-rdkit-regression", model_type=ModelType.REGRESSOR
        )
        features_to_model.set_output_tags(["aqsol", "regression", "rdkit"])
        features_to_model.transform(
            target_column="solubility", description="AQSol/RDKit Regression Model", feature_list=feature_list
        )

    # Create the aqsol regression Endpoint
    if recreate or not Endpoint("aqsol-rdkit-regression-end").exists():
        model_to_endpoint = ModelToEndpoint("aqsol-rdkit-regression", "aqsol-rdkit-regression-end", serverless=False)
        model_to_endpoint.set_output_tags(["aqsol", "rdkit", "regression"])
        model_to_endpoint.transform()
