"""Launch an AWS Inference Run"""

import os
import sys
import time
import pandas as pd
import logging
import boto3
import sagemaker

# Workbench Imports
os.environ["WORKBENCH_SKIP_LOGGING"] = "True"  # For extra speed :p
from workbench.utils.fast_inference import fast_inference


# Set up logging
log = logging.getLogger()


def get_sagemaker_session() -> sagemaker.Session:

    # Create a SageMaker session
    session = sagemaker.Session()

    # Get the SageMaker role
    role = "Workbench-ExecutionRole"

    # Attach the role to the session
    boto3.client("sts").assume_role(
        RoleArn=f'arn:aws:iam::{session.boto_session.client("sts").get_caller_identity()["Account"]}:role/{role}',
        RoleSessionName="WorkbenchSession",
    )

    return session


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

    # Get out Sagemaker Session
    sm_session = get_sagemaker_session()

    # Local data this will duplicate a launch from an App like LiveDesign/StarDrop
    data = pd.read_csv("test_evaluation_data.csv")

    # Run the inference on the Endpoint and print out the results
    data_sample = data.sample(n=num_rows, replace=True)
    print(f"\nTiming Inference on {len(data_sample)} rows")
    start_time = time.time()
    results = fast_inference(test_endpoint_name, data_sample, sm_session)
    inference_time = time.time() - start_time
    print(f"Inference Time: {inference_time} on Endpoint: {test_endpoint_name}")

    # Print out the results
    print("\nInference Results:")
    print(results)
