"""Tests for the Data to Data (light) Transforms"""

# Local imports
from sageworks.transforms.data_to_data.light.data_to_data_light import DataToDataLight


def test():
    """Tests for the Data to Data (light) Transforms"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = 'abalone_data'
    output_uuid = 'abalone_data_copy'
    DataToDataLight(input_uuid, output_uuid).transform()


if __name__ == "__main__":
    test()
