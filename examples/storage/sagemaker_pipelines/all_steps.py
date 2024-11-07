import sageworks
from sageworks.api.data_source import DataSource
from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model, ModelType


print(f"Hello from SageWorks: {sageworks.__version__}")

# Create the abalone_data DataSource
ds = DataSource("s3://sageworks-public-data/common/abalone.csv", name="sp_abalone_data")

# Now create a FeatureSet
ds.to_features("sp_abalone_features", id_column="id")

# Create the abalone_regression Model
fs = FeatureSet("sp_abalone_features")
fs.to_model(
    ModelType.REGRESSOR,
    name="sp-abalone-regression",
    target_column="class_number_of_rings",
    tags=["abalone", "regression"],
    description="Abalone Regression Model",
)

# Create the abalone_regression Endpoint
model = Model("sp-abalone-regression")
model.to_endpoint(name="sp-abalone-regression-end", tags=["abalone", "regression"])
