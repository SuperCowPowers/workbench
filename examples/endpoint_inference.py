from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model
from sageworks.api.endpoint import Endpoint

# Grab the abalone regression Model
model = Model("abalone-regression")

# By default, an Endpoint is serverless, but you can make it non-serverless
serverless = True
model.to_endpoint(name="abalone-regression-end", tags=["abalone", "regression"], serverless=serverless)

# Now we'll run inference on the endpoint
endpoint = Endpoint("abalone-regression-end")

# SageWorks has full ML Pipeline provenance, so we can backtrack the inputs,
# get a DataFrame of data (not used for training) and run inference
fs_name = model.get_input()
fs = FeatureSet(fs_name)
athena_table = fs.get_training_view_table()
df = fs.query(f"SELECT * FROM {athena_table} where training = 0")
results = endpoint.predict(df)
print(results[["class_number_of_rings", "prediction"]])
