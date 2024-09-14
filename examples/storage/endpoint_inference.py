"""Example script of running inference on an Endpoint"""

import argparse
import pandas as pd

# SageWorks Imports
from sageworks.core.artifacts.feature_set_core import FeatureSetCore
from sageworks.core.artifacts.model_core import ModelCore
from sageworks.core.artifacts.endpoint_core import EndpointCore


def run_inference(endpoint_name):
    # Set options for actually seeing the dataframe
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # Grab the Endpoint
    my_endpoint = EndpointCore(endpoint_name)

    # Grab the FeatureSet by backtracking from the Endpoint
    model = my_endpoint.get_input()
    feature_set = ModelCore(model).get_input()
    features = FeatureSetCore(feature_set)
    table = features.view("training").table
    test_df = features.query(f"SELECT * FROM {table} where training = 0")

    # Drop some columns
    test_df.drop(["write_time", "api_invocation_time", "is_deleted"], axis=1, inplace=True)

    # Make predictions on the Endpoint
    pred_df = my_endpoint.predict(test_df[:10])
    print(pred_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a SageWorks Endpoint")
    parser.add_argument("endpoint_name", type=str, help="Name of the SageWorks Endpoint")
    args = parser.parse_args()

    run_inference(args.endpoint_name)
