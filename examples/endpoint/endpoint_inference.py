from workbench.api import Endpoint
from workbench.utils.endpoint_utils import get_evaluation_data

# Grab an existing Endpoint
endpoint = Endpoint("abalone-regression")

# Workbench has full ML Pipeline provenance, so we can backtrack the inputs,
# get a DataFrame of data (not used for training) and run inference
df = get_evaluation_data(endpoint)

# Run inference/predictions on the Endpoint
results_df = endpoint.inference(df)

# Run inference/predictions and capture the results
results_df = endpoint.inference(df, capture_name="test_inference")

# Run inference/predictions using the FeatureSet evaluation data
results_df = endpoint.auto_inference()
