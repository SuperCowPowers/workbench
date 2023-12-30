from sageworks.api.endpoint import Endpoint
from pprint import pprint

# Grab an existing Endpoint and print out it's details
endpoint = Endpoint("abalone-regression-end")
pprint(endpoint.details())
