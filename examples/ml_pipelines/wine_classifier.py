"""This Script creates a Wine Classification ML Pipeline

DataSources:
    - wine_data (using this to create the FeatureSet)
FeatureSets:
    - wine_feature_set
Models:
    - wine-classification
Endpoints:
    - wine-classification-end
"""

from workbench.api import DataSource, FeatureSet, Model, ModelType

if __name__ == "__main__":

    # Create the wine_features FeatureSet
    ds = DataSource("wine_data")
    fs = ds.to_features("wine_features", tags=["wine", "classification"])
    fs.set_owner("test")

    # Create the wine classification Model
    fs = FeatureSet("wine_features")
    m = fs.to_model(
        name="wine-classification",
        model_type=ModelType.CLASSIFIER,
        target_column="wine_class",
        tags=["wine", "classification"],
        description="Wine Classification Model",
    )
    m.set_owner("test")
    m.set_class_labels(["TypeA", "TypeB", "TypeC"])

    # Create the wine classification Endpoint
    m = Model("wine-classification")
    end = m.to_endpoint("wine-classification", tags=["wine", "classification"])

    # Run inference on the endpoint
    end.auto_inference()
    end.set_owner("test")
