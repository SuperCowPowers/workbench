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

from sageworks.api.data_source import DataSource
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType
from sageworks.api.endpoint import Endpoint

from sageworks.core.transforms.data_to_features.light.rdkit_descriptors import RDKitDescriptors
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker

if __name__ == "__main__":
    # This forces a refresh on all the data we get from the AWs Broker
    AWSServiceBroker().get_all_metadata(force_refresh=True)

    # Get the path to the dataset in S3
    s3_path = "s3://sageworks-public-data/comp_chem/aqsol_public_data.csv"

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the aqsol_data DataSource
    if recreate or not DataSource("aqsol_data").exists():
        DataSource(s3_path, name="aqsol_data")

    # Create the aqsol_features FeatureSet
    if recreate or not FeatureSet("aqsol_features").exists():
        ds = DataSource("aqsol_data")
        ds.to_features("aqsol_features", id_column="id")

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
    # RDKIT Descriptor Artifacts
    #
    # Create the rdkit FeatureSet (this is an example of using lower level classes)
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
        feature_set.to_model(
            ModelType.REGRESSOR,
            target_column="solubility",
            name="aqsol-rdkit-regression",
            feature_list=feature_list,
            description="AQSol/RDKit Regression Model",
            tags=["aqsol", "regression", "rdkit"],
        )

    # Create the aqsol regression Endpoint
    if recreate or not Endpoint("aqsol-rdkit-regression-end").exists():
        m = Model("aqsol-rdkit-regression")
        m.to_endpoint(name="aqsol-rdkit-regression-end", tags=["aqsol", "regression", "rdkit"])
