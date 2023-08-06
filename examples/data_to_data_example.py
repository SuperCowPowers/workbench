"""DataToDataExample: Example Class that demonstrates a Data Source to Data Source Transform"""

# Local imports
from sageworks.transforms.data_to_data.light.data_to_data_light import DataToDataLight
from sageworks.utils import pandas_utils


class DataToDataExample(DataToDataLight):
    def __init__(self, input_uuid: str, output_uuid: str):
        """DataToDataExample: Example Class that demonstrates a Data Source to Data Source Transform"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

    def transform_impl(self, drop_na="any"):
        """Just dropping NaNs, but you could do anything that you want, simply take the
        input dataframe and produce an output dataframe (of any form)"""

        # Drop Rows that have NaNs in them
        self.output_df = pandas_utils.drop_nans(self.input_df, how=drop_na)


# Simple test of the DataToDataExample functionality
def test():
    """Test the DataToDataExample Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "test_data"
    output_uuid = "test_data_clean"
    DataToDataExample(input_uuid, output_uuid).transform(drop_na="any")


if __name__ == "__main__":
    test()
