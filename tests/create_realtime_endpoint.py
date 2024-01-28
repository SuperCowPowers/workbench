"""This Script creates a Realtime Endpoint for testing

Endpoints:
    - abalone-regression-end-rt
"""

import logging

# Sageworks Imports
from sageworks.api.model import Model
from sageworks.api.endpoint import Endpoint
from sageworks.aws_service_broker.aws_service_broker import AWSServiceBroker

# Setup the logger
log = logging.getLogger("sageworks")

if __name__ == "__main__":
    # This forces a refresh on all the data we get from the AWS Broker
    AWSServiceBroker().get_all_metadata(force_refresh=True)

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the abalone_regression Endpoint
    if recreate or not Endpoint("abalone-regression-end-rt").exists():
        m = Model("abalone-regression")
        m.to_endpoint("abalone-regression-end-rt", tags=["abalone", "regression", "realtime"], serverless=False)
