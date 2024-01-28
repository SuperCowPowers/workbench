"""Tests for the Data to Data (light) Transforms"""

import pytest

# Local imports
from sageworks.core.transforms.data_to_data.light.data_to_data_light import DataToDataLight


@pytest.mark.long
def test():
    """Tests for the Data to Data (light) Transforms"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "abalone_data"
    output_uuid = "abalone_data_copy"
    data_to_data = DataToDataLight(input_uuid, output_uuid)
    data_to_data.set_output_tags(["abalone", "public"])
    data_to_data.transform()


if __name__ == "__main__":
    test()
