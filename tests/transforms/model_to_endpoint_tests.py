"""Tests for the Model to Endpoint Transforms"""

import pytest

# Local Imports
from workbench.core.transforms.model_to_endpoint.model_to_endpoint import ModelToEndpoint
from workbench.api.endpoint import Endpoint


# Simple test of the ModelToEndpoint functionality
@pytest.mark.long
def test():
    """Tests for the Model to Endpoint Transforms"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "abalone-regression"
    output_uuid = "abalone-regression"
    to_endpoint = ModelToEndpoint(input_uuid, output_uuid)
    to_endpoint.set_output_tags(["abalone", "public"])
    to_endpoint.transform()

    # Now run inference on the endpoint
    endpoint = Endpoint(output_uuid)
    endpoint.auto_inference()


if __name__ == "__main__":
    test()
