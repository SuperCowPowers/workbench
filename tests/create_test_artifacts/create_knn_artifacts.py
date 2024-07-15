"""This Script creates the Abalone and AQSol KNN Model Artifacts in AWS/SageWorks

Models:
    - abalone-knn-reg
    - aqsol-knn-reg
Endpoints:
    - abalone-knn-end
    - aqsol-knn-end
"""

import logging
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType
from sageworks.api.endpoint import Endpoint
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker

log = logging.getLogger("sageworks")


if __name__ == "__main__":
    # This forces a refresh on all the data we get from the AWs Broker
    AWSServiceBroker().get_all_metadata(force_refresh=True)

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the Abalone KNN Regression Model
    if recreate or not Model("abalone-knn-reg").exists():

        # Transform FeatureSet into KNN Regression Model
        feature_set = FeatureSet("abalone_features")
        feature_set.to_model(
            model_class="KNeighborsRegressor",
            target_column="class_number_of_rings",
            name="abalone-knn-reg",
            description="Abalone KNN Regression",
            tags=["abalone", "knn"],
            train_all_data=True,
        )

    # Create the Abalone KNN Regression Endpoint
    if recreate or not Endpoint("abalone-knn-end").exists():
        m = Model("abalone-knn-reg")
        end = m.to_endpoint(name="abalone-knn-end", tags=["abalone", "knn"])

        # Run auto-inference on the Abalone KNN Regression Endpoint
        end.auto_inference(capture=True)

    # Create the AQSol KNN Regression Model
    if recreate or not Model("aqsol-knn-reg").exists():

        # AQSol Features
        features = [
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

        # Transform FeatureSet into KNN Regression Model
        feature_set = FeatureSet("aqsol_features")
        feature_set.to_model(
            ModelType.REGRESSOR,
            model_class="KNeighborsRegressor",
            target_column="solubility",
            feature_list=features,
            name="aqsol-knn-reg",
            description="AQSol KNN Regression",
            tags=["aqsol", "knn"],
            train_all_data=True,
        )

    # Create the AQSol KNN Regression Endpoint
    if recreate or not Endpoint("aqsol-knn-end").exists():
        m = Model("aqsol-knn-reg")
        end = m.to_endpoint(name="aqsol-knn-end", tags=["aqsol", "knn"])

        # Run auto-inference on the AQSol Quantile Regression Endpoint
        end.auto_inference(capture=True)
