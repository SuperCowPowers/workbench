"""This Script creates the Abalone and AQSol KNN Model Artifacts in AWS/Workbench

Models:
    - abalone-knn-reg
    - aqsol-knn-reg
Endpoints:
    - abalone-knn-end
    - aqsol-knn-end
"""

import logging
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType
from workbench.api.endpoint import Endpoint

log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the Abalone KNN Regression Model
    if recreate or not Model("abalone-knn-reg").exists():

        # Transform FeatureSet into KNN Regression Model
        feature_set = FeatureSet("abalone_features")
        feature_set.to_model(
            name="abalone-knn-reg",
            model_type=ModelType.REGRESSOR,
            target_column="class_number_of_rings",
            description="Abalone KNN Regression",
            tags=["abalone", "knn"],
            scikit_model_class="KNeighborsRegressor",
            model_import_str="from sklearn.neighbors import KNeighborsRegressor",
            train_all_data=True,
        )

    # Create the Abalone KNN Regression Endpoint
    if recreate or not Endpoint("abalone-knn").exists():
        m = Model("abalone-knn-reg")
        end = m.to_endpoint(name="abalone-knn", tags=["abalone", "knn"])

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
            name="aqsol-knn-reg",
            model_type=ModelType.REGRESSOR,
            scikit_model_class="KNeighborsRegressor",
            model_import_str="from sklearn.neighbors import KNeighborsRegressor",
            target_column="solubility",
            feature_list=features,
            description="AQSol KNN Regression",
            tags=["aqsol", "knn"],
            train_all_data=True,
        )

    # Create the AQSol KNN Regression Endpoint
    if recreate or not Endpoint("aqsol-knn").exists():
        m = Model("aqsol-knn-reg")
        end = m.to_endpoint(name="aqsol-knn", tags=["aqsol", "knn"])

        # Run auto-inference on the AQSol Quantile Regression Endpoint
        end.auto_inference(capture=True)
