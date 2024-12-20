"""Script that loops through all endpoints and checks if they are ready"""

import logging

# Workbench Imports
from workbench.api import Meta, Endpoint

# Setup logging
log = logging.getLogger("workbench")

# Get all the endpoints
endpoints = Meta().endpoints()
for end_name in endpoints["Name"]:
    end = Endpoint(end_name)
    if end.ready():
        log.important(f"Endpoint {end_name} is ready!")
    else:
        log.important(f"Endpoint {end_name} is not ready...Calling onboard.... ")
        end.onboard()
