"""DataSourceToDF: Class to transform a Data Source into a Pandas DataFrame"""
import logging
import pandas as pd

# Local imports
from sageworks.utils.logging import logging_setup
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.aws_artifacts.data_sources.athena_source import AthenaSource

# Setup Logging
logging_setup()


class DataSourceToDF(Transform):
    def __init__(self):
        """DataSourceToDF: Class to transform a Data Source into a Pandas DataFrame"""

        # Set up all my class instance vars
        self.log = logging.getLogger(__name__)
        self.input_uuid = None
        self.input_data_source = None
        self.output_df = None
        self.data_catalog_db = 'sageworks'

    def input_type(self) -> TransformInput:
        """What Input Type does this Transform Consume"""
        return TransformInput.DATA_SOURCE

    def output_type(self) -> TransformOutput:
        """What Output Type does this Transform Produce"""
        return TransformOutput.PANDAS_DF

    def set_input_uuid(self, input_uuid: str):
        self.input_uuid = input_uuid

    def set_output_uuid(self, uuid: str):
        """Not Implemented: Just satisfying the Transform abstract method requirements"""
        pass

    def get_output(self) -> pd.DataFrame:
        """Get the DataFrame Output from this Transform"""
        return self.output_df

    def validate_input(self) -> bool:
        """Validate the Input for this Transform"""
        self.input_data_source = AthenaSource(self.data_catalog_db, self.input_uuid)
        return self.input_data_source.check()

    def transform(self, overwrite: bool = True):
        """Convert the DataSource into a Pandas DataFrame"""
        self.output_df = self.input_data_source.query(f"select * from {self.input_uuid}")


# Simple test of the DataSourceToDF functionality
def test():
    """Test the DataSourceToDF Class"""

    # Setup Pandas output options
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    # Grab a Data Source
    data_uuid = 'aqsol_data'

    # Create the DataSource to DF Transform
    data_to_df = DataSourceToDF()
    data_to_df.set_input_uuid(data_uuid)

    # Is my input data AOK?
    assert(data_to_df.validate_input())

    # Transform the DataSource into a Pandas DataFrame
    data_to_df.transform()

    # Grab the output and show it
    my_df = data_to_df.get_output()
    print(my_df)


if __name__ == "__main__":
    test()
