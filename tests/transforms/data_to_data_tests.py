"""Tests for the Data to Data (light) Transforms"""

import pytest

# Local imports
from workbench.core.transforms.data_to_data.light.data_to_data_light import DataToDataLight


@pytest.mark.long
def test():
    """Tests for the Data to Data (light) Transforms"""

    # Create the class with inputs and outputs and invoke the transform
    input_name = "abalone_data"
    output_name = "abalone_long_tags"
    data_to_data = DataToDataLight(input_name, output_name)
    tags = ["test", "public"]
    data_to_data.set_output_tags(tags)
    data_to_data.transform()


if __name__ == "__main__":
    test()
