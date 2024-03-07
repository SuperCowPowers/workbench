from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import ModelType


# Create the abalone_regression Model
fs = FeatureSet("sp_abalone_features")
fs.to_model(
    ModelType.REGRESSOR,
    name="sp-abalone-regression",
    target_column="class_number_of_rings",
    tags=["abalone", "regression"],
    description="Abalone Regression Model",
)
