"""FeaturesToPandas: Class to transform a Data Source into a Spark DataFrame"""
import logging
import pandas as pd

# Local imports
from sageworks.utils.logging import logging_setup
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.artifacts.data_sources.athena_source import AthenaSource

# Setup Logging
logging_setup()


class FeaturesToPandas(Transform):
    def __init__(self):
        """FeaturesToPandas: Class to transform a Data Source into a Spark DataFrame"""

        # Call superclass init
        super().__init__()

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

    def transform(self, max_rows=100000):
        """Convert the DataSource into a Pandas DataFrame"""

        # Validate our input and get the number of rows in the DataSource
        self.validate_input()
        num_rows = self.input_data_source.num_rows()

        # If the data source has more rows than max_rows, do a sample query
        if num_rows > max_rows:
            percentage = round(max_rows*100.0/num_rows)
            self.log.warning(f"DataSource has {num_rows} rows.. sampling down to {max_rows}...")
            query = f"SELECT * FROM {self.input_uuid} TABLESAMPLE BERNOULLI({percentage})"
        else:
            query = f"SELECT * FROM {self.input_uuid}"
        self.output_df = self.input_data_source.query(query)


# Simple test of the FeaturesToPandas functionality
def test():
    """Test the FeaturesToPandas Class"""

    # Setup Pandas output options
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    # Grab a Data Source
    data_uuid = 'aqsol_data'

    # Create the DataSource to DF Transform
    data_to_df = FeaturesToPandas()
    data_to_df.set_input_uuid(data_uuid)

    # Transform the DataSource into a Pandas DataFrame (with max_rows = 1000)
    data_to_df.transform(max_rows=1000)

    # Grab the output and show it
    my_df = data_to_df.get_output()
    print(my_df)


if __name__ == "__main__":
    test()
