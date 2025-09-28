"""This Script creates a Realtime Endpoint for testing

Endpoints:
    - abalone-regression-end-rt
"""

import logging

# Workbench Imports
from workbench.api.model import Model
from workbench.api.endpoint import Endpoint

# Setup the logger
log = logging.getLogger("workbench")

if __name__ == "__main__":

    # Recreate Flag in case you want to recreate the artifacts
    recreate = False

    # Create the abalone_regression Endpoint
    if recreate or not Endpoint("abalone-regression-rt").exists():
        m = Model("abalone-regression")
        m.to_endpoint("abalone-regression-rt", tags=["abalone", "regression", "realtime"], serverless=False)
