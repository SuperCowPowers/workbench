"""DFToDataSource: Class to publish a Pandas DataFrame into Athena"""
import awswrangler as wr
import json
import logging
import pandas as pd

# Local imports
from sageworks.utils.logging import logging_setup
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.aws_artifacts.data_sources.athena_source import AthenaSource

# Setup Logging
logging_setup()


class DFToDataSource(Transform):
    def __init__(self):
        """DFToDataSource: Class to publish Pandas DataFrames into Athena"""

        # Set up all my class instance vars
        self.log = logging.getLogger(__name__)
        self.input_df = None
        self.output_uuid = None
        self.data_catalog_db = 'sageworks'
        self.data_source_s3_path = 's3://sageworks-data-sources'

    def input_type(self) -> TransformInput:
        """What Input Type does this Transform Consume"""
        return TransformInput.PANDAS_DF

    def output_type(self) -> TransformOutput:
        """What Output Type does this Transform Produce"""
        return TransformOutput.DATA_SOURCE

    def set_input(self, input_df: pd.DataFrame):
        """Set the Input DataFrame for this Transform"""
        self.input_df = input_df

    def set_input_uuid(self, uuid: str):
        """Not Implemented: Just satisfying the Transform abstract method requirements"""
        pass

    def set_output_uuid(self, uuid: str):
        """Set the Name for the output Data Source"""
        self.output_uuid = uuid

    def get_output(self) -> AthenaSource:
        """Get the Output from this Transform"""
        return AthenaSource(self.data_catalog_db, self.output_uuid)

    def validate_input(self) -> bool:
        """Validate the Input for this Transform"""

        # Simple check on the input dataframe
        return self.input_df is not None and not self.input_df.empty

    def transform(self, overwrite: bool = True):
        """Convert the Pandas DataFrame into Parquet Format in the SageWorks S3 Bucket, and
           store the information about the data to the AWS Data Catalog sageworks database"""

        # Add some tags here
        tags = ['sageworks', 'public']

        # Create the Output Parquet file S3 Storage Path
        s3_storage_path = f"{self.data_source_s3_path}/{self.output_uuid}"

        # Write out the DataFrame to Parquet/DataStore/Athena
        wr.s3.to_parquet(self.input_df, path=s3_storage_path, dataset=True, mode='overwrite',
                         database=self.data_catalog_db, table=self.output_uuid,
                         description=f'SageWorks data source: {self.output_uuid}',
                         filename_prefix=f'{self.output_uuid}_',
                         parameters={'tags': json.dumps(tags)},
                         partition_cols=None)  # FIXME: Have some logic around partition columns


# Simple test of the DFToDataSource functionality
def test():
    """Test the DFToDataSource Class"""
    from datetime import datetime

    # Setup Pandas output options
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    # Create some fake data
    fake_df = pd.DataFrame({'name': ['sue'], 'age': [41], 'score': [7.8], 'date': [datetime.now()]})

    # Create my Data Loader
    output_uuid = 'test_data'
    my_loader = DFToDataSource()
    my_loader.set_input(fake_df)
    my_loader.set_output_uuid(output_uuid)

    # Does my data pass validation?
    assert(my_loader.validate_input())

    # Store this data into Athena/SageWorks
    my_loader.transform()

    # Grab the output and query it for a dataframe
    output = my_loader.get_output()
    query = f"select * from {output_uuid} limit 5"
    df = output.query(query)

    # Show the dataframe
    print(df)


if __name__ == "__main__":
    test()
