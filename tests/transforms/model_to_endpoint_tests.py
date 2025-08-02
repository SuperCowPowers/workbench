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
    input_name = "abalone-regression"
    output_name = "abalone-regression-temp"
    to_endpoint = ModelToEndpoint(input_name, output_name)
    to_endpoint.set_output_tags(["temp", "abalone", "public"])
    to_endpoint.transform()

    # Now run inference on the endpoint
    endpoint = Endpoint(output_name)
    endpoint.auto_inference()


if __name__ == "__main__":
    test()
