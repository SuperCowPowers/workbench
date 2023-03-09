"""CleanData: Example Class that demonstrates data cleanup for Light DataSources using Pandas"""

# Local imports
from sageworks.utils.logging import logging_setup
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.transforms.transform_utils.data_to_pandas import DataToPandas
from sageworks.transforms.transform_utils.pandas_to_data import PandasToData

# Setup Logging
logging_setup()


class CleanData(Transform):
    def __init__(self, input_uuid: str = None, output_uuid: str = None):
        """CleanData: Class for filtering, sub-setting, and value constraints on Light DataSources uses Pandas"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.DATA_SOURCE
        self.output_type = TransformOutput.DATA_SOURCE

    def transform(self):
        """Pull the input DataSource make sure it's 'clean' and output to a DataSource"""

        # Grab the Input (Data Source)
        input_df = DataToPandas(self.input_uuid).get_output()  # Shorthand for transform, get_output

        # Cleanup up the data
        input_df = input_df

        # Now publish to the output location
        output_data = PandasToData(self.output_uuid)
        output_data.set_input(input_df)
        output_data.transform()


# Simple test of the CleanData functionality
def test():
    """Test the CleanData Class"""
    import pandas as pd
    from sageworks.artifacts.data_sources.athena_source import AthenaSource

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = 'aqsol_data'
    output_uuid = 'aqsol_data_clean'
    CleanData(input_uuid, output_uuid).transform()

    # Grab the output and query it for a dataframe
    output = AthenaSource(output_uuid)
    df = output.query(f"select * from {output_uuid} limit 5")

    # Setup Pandas output options
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    # Show the dataframe
    print(df)


if __name__ == "__main__":
    test()
