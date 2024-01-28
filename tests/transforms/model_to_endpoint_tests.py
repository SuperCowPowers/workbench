"""Tests for the Model to Endpoint Transforms"""

import pytest

# Local Imports
from sageworks.core.transforms.model_to_endpoint.model_to_endpoint import ModelToEndpoint


# Simple test of the ModelToEndpoint functionality
@pytest.mark.long
def test():
    """Tests for the Model to Endpoint Transforms"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "abalone-regression"
    output_uuid = "abalone-regression-end"
    to_endpoint = ModelToEndpoint(input_uuid, output_uuid)
    to_endpoint.set_output_tags(["abalone", "public"])
    to_endpoint.transform()


if __name__ == "__main__":
    test()
