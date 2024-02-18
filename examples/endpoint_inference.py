from sageworks.api.feature_set import FeatureSet
from sageworks.api.model import Model
from sageworks.api.endpoint import Endpoint

# Grab an existing Endpoint
endpoint = Endpoint("abalone-regression-end")

# SageWorks has full ML Pipeline provenance, so we can backtrack the inputs,
# get a DataFrame of data (not used for training) and run inference
model = Model(endpoint.get_input())
fs = FeatureSet(model.get_input())
athena_table = fs.get_training_view_table()
df = fs.query(f"SELECT * FROM {athena_table} where training = 0")

# Run inference/predictions on the Endpoint
results = endpoint.inference(df)
print(results[["class_number_of_rings", "prediction"]])
