"""Launch an AWS Inference Run"""

import os
import sys
import time
import pandas as pd
import logging

# Workbench-Bridge Imports
from workbench_bridges.endpoints.fast_inference import fast_inference


# Set up logging
log = logging.getLogger()


def download_data(endpoint_name: str):
    """Download the data Workbench FeatureSet

    Args:
        endpoint_name (str): The name of the Endpoint
    """
    from workbench.api import FeatureSet, Model, Endpoint

    fs = FeatureSet(Model(Endpoint(endpoint_name).get_input()).get_input())
    df = fs.pull_dataframe()
    df.to_csv("test_evaluation_data.csv", index=False)


if __name__ == "__main__":

    # Endpoint name to test
    test_endpoint_name = "test-timing-realtime"
    num_rows = 1000

    # Check if we have local data
    if not os.path.exists("test_evaluation_data.csv"):
        log.warning("Downloading Data... Rerun the script after the download completes")
        download_data(test_endpoint_name)
        sys.exit(1)

    # Local data this will duplicate a launch from an App like LiveDesign/StarDrop
    data = pd.read_csv("test_evaluation_data.csv")

    # Run the inference on the Endpoint and print out the results
    data_sample = data.sample(n=num_rows, replace=True)
    print(f"\nTiming Inference on {len(data_sample)} rows")
    start_time = time.time()
    results = fast_inference(test_endpoint_name, data_sample)
    inference_time = time.time() - start_time
    print(f"Inference Time: {inference_time} on Endpoint: {test_endpoint_name}")

    # Print out the results
    print("\nInference Results:")
    print(results)
