"""Create the AQSol solubility classification Model and Endpoint.

Consumes the `fs:aqsol_features` FeatureSet produced by aqsol_feature_set.py.

Models:
    - aqsol-class
Endpoints:
    - aqsol-class
"""

from workbench.api import FeatureSet, Model, ModelType, ModelFramework

FEATURE_LIST = [
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

if __name__ == "__main__":

    # Build the classification Model from the aqsol_features FeatureSet
    fs = FeatureSet("aqsol_features")
    m = fs.to_model(
        name="aqsol-class",
        model_type=ModelType.CLASSIFIER,
        model_framework=ModelFramework.XGBOOST,
        target_column="solubility_class",
        feature_list=FEATURE_LIST,
        description="AQSol Classification Model",
        tags=["aqsol", "classification"],
    )
    m.set_owner("test")
    m.set_class_labels(["low", "medium", "high"])

    # Deploy the classification Endpoint
    m = Model("aqsol-class")
    end = m.to_endpoint(tags=["aqsol", "classification"])
    end.set_owner("test")
    end.test_inference()
