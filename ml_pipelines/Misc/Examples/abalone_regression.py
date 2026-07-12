"""This Script creates an Abalone Regression ML Pipeline

DataSources:
    - abalone_data (using this to create the FeatureSet)
FeatureSets:
    - abalone_features
Models:
    - abalone-regression
Endpoints:
    - abalone-regression-end
"""

from workbench.api import DataSource, FeatureSet, Model, ModelType, ModelFramework

if __name__ == "__main__":

    # Create the abalone_features FeatureSet
    ds = DataSource("abalone_data")
    fs = ds.to_features("abalone_features", tags=["abalone", "regression"])
    fs.set_owner("test")

    # Create the abalone regression Model (predict rings/age from physical measurements)
    fs = FeatureSet("abalone_features")
    m = fs.to_model(
        name="abalone-regression",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="rings",
        tags=["abalone", "regression"],
        description="Abalone Regression Model",
    )
    m.set_owner("test")

    # Create the abalone regression Endpoint
    m = Model("abalone-regression")
    end = m.to_endpoint("abalone-regression", tags=["abalone", "regression"])

    # Run inference on the endpoint
    end.test_inference()
    end.cross_fold_inference()
    end.set_owner("test")
