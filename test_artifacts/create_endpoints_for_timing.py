"""This Script creates Endpoints for Timing Tests

Models:
    - Using abalone-regression to create the two endpoints.
Endpoints:
    - test-timing-serverless, test-timing-realtime
"""

import sys
import logging

# Workbench imports
from workbench.api import Model, Endpoint

log = logging.getLogger("workbench")


if __name__ == "__main__":

    # Grab our Model
    model = Model("abalone-regression")
    if not model.exists():
        log.error("The abalone-regression model does not exist. Please create it first.")
        sys.exit(1)

    # Create the two Endpoints
    if not Endpoint("test-timing-serverless").exists():
        end = model.to_endpoint("test-timing-serverless", tags=["test", "timing"])

    if not Endpoint("test-timing-realtime").exists():
        end = model.to_endpoint("test-timing-realtime", tags=["test", "timing"], serverless=False)
