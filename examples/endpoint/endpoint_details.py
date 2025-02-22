from workbench.api.endpoint import Endpoint
from pprint import pprint

# Grab an existing Endpoint and print out it's details
endpoint = Endpoint("abalone-regression")
pprint(endpoint.details())
