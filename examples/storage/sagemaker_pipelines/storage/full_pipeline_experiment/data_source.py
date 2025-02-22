import workbench
from workbench.api.data_source import DataSource
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model, ModelType


print(f"Hello from Workbench: {workbench.__version__}")

# Create the abalone_data DataSource
ds = DataSource("s3://workbench-public-data/common/abalone.csv", name="sp_abalone_data")

# Now create a FeatureSet
ds.to_features("sp_abalone_features")

# Create the abalone_regression Model
fs = FeatureSet("sp_abalone_features")
fs.to_model(
    name="sp-abalone-regression",
    model_type=ModelType.REGRESSOR,
    target_column="class_number_of_rings",
    tags=["abalone", "regression"],
    description="Abalone Regression Model",
)

# Create the abalone_regression Endpoint
model = Model("sp-abalone-regression")
model.to_endpoint(name="sp-abalone-regression", tags=["abalone", "regression"])
