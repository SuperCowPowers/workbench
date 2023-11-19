"""This Script creates a Realtime Endpoint for testing

Endpoints:
    - abalone-regression-end-rt
"""
import logging

# Sageworks Imports
from sageworks.artifacts.endpoints.endpoint import Endpoint
from sageworks.transforms.model_to_endpoint.model_to_endpoint import ModelToEndpoint
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker

# Setup the logger
log = logging.getLogger("sageworks")

if __name__ == "__main__":
    # This forces a refresh on all the data we get from the AWs Broker
    AWSServiceBroker().get_all_metadata(force_refresh=True)

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the abalone_regression Endpoint
    if recreate or not Endpoint("abalone-regression-end-rt").exists():
        model_to_endpoint = ModelToEndpoint("abalone-regression", "abalone-regression-end-rt", serverless=False)
        model_to_endpoint.set_output_tags(["abalone", "regression", "realtime"])
        model_to_endpoint.transform()
