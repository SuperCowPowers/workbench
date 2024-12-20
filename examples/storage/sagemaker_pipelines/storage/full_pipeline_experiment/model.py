from workbench.api.feature_set import FeatureSet
from workbench.api.model import ModelType


# Create the abalone_regression Model
fs = FeatureSet("sp_abalone_features")
fs.to_model(
    name="sp-abalone-regression",
    model_type=ModelType.REGRESSOR,
    target_column="class_number_of_rings",
    tags=["abalone", "regression"],
    description="Abalone Regression Model",
)
