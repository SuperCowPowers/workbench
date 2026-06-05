"""Create the AQSol solubility regression Model and Endpoint.

Consumes the `fs:aqsol_features` FeatureSet produced by aqsol_feature_set.py.

Models:
    - aqsol-regression
Endpoints:
    - aqsol-regression
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

    # Build the regression Model from the aqsol_features FeatureSet
    fs = FeatureSet("aqsol_features")
    m = fs.to_model(
        name="aqsol-regression",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="solubility",
        feature_list=FEATURE_LIST,
        description="AQSol Regression Model",
        tags=["aqsol", "regression"],
    )
    m.set_owner("test")

    # Deploy the regression Endpoint
    m = Model("aqsol-regression")
    end = m.to_endpoint(tags=["aqsol", "regression"])
    end.set_owner("test")
    end.test_inference()
