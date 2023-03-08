"""S3LightToAthena: Class to move LIGHT S3 Files into Athena"""
import awswrangler as wr
import json
import logging

# Local imports
from sageworks.utils.logging import logging_setup
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.artifacts.data_sources.athena_source import AthenaSource

# Setup Logging
logging_setup()


class S3LightToAthena(Transform):
    def __init__(self):
        """S3LightToAthena: Class to move LIGHT S3 Files into Athena"""

        # Set up all my class instance vars
        self.log = logging.getLogger(__name__)
        self.s3_file_path = None
        self.output_uuid = None
        self.data_catalog_db = 'sageworks'
        self.data_source_s3_path = 's3://sageworks-data-sources'

    def input_type(self) -> TransformInput:
        """What Input Type does this Transform Consume"""
        return TransformInput.S3_OBJECT

    def output_type(self) -> TransformOutput:
        """What Output Type does this Transform Produce"""
        return TransformOutput.DATA_SOURCE

    def set_input_uuid(self, s3_file_path: str):
        """Set the Input for this Transform (S3 Path for this Transform)"""
        self.s3_file_path = s3_file_path

    def set_output_uuid(self, uuid: str):
        """Set the Name for the output Data Source"""
        self.output_uuid = uuid

    def get_output(self) -> AthenaSource:
        """Get the Output from this Transform"""
        return AthenaSource(self.data_catalog_db, self.output_uuid)

    def validate_input(self) -> bool:
        """Validate the Input for this Transform"""

        # Does the S3 object exist and can it be accessed
        print(f'Input Validation {self.s3_file_path}...')
        return wr.s3.does_object_exist(self.s3_file_path)

    def input_size_mb(self) -> int:
        """ Get the size of the input S3 object in MBytes"""
        size_in_bytes = wr.s3.size_objects(self.s3_file_path)[self.s3_file_path]
        size_in_mb = round(size_in_bytes/1_000_000)
        return size_in_mb

    def transform(self, overwrite: bool = True):
        """Convert the CSV data into Parquet Format in the SageWorks S3 Bucket, and
           store the information about the data to the AWS Data Catalog sageworks database"""

        # Sanity Check for S3 Object size
        object_megabytes = self.input_size_mb()
        if object_megabytes > 100:
            self.log.error(f"S3 Object too big ({object_megabytes} MBytes): Use the S3BigToAthena class!")
            return

        # Add some tags here
        tags = ['sageworks', 'public']

        # Read in the S3 CSV as a Pandas DataFrame
        df = wr.s3.read_csv(self.s3_file_path, low_memory=False)

        # Create the Output Parquet file S3 Storage Path
        s3_storage_path = f"{self.data_source_s3_path}/{self.output_uuid}"

        # Write out the DataFrame to Parquet/DataStore/Athena
        wr.s3.to_parquet(df, path=s3_storage_path, dataset=True, mode='overwrite',
                         database=self.data_catalog_db, table=self.output_uuid,
                         description=f'SageWorks data source: {self.output_uuid}',
                         filename_prefix=f'{self.output_uuid}_',
                         parameters={'tags': json.dumps(tags)},
                         partition_cols=None)  # FIXME: Have some logic around partition columns


# Simple test of the S3LightToAthena functionality
def test():
    """Test the S3LightToAthena Class"""
    import pandas as pd

    # Create my Data Loader
    output_uuid = 'aqsol_data'
    my_loader = S3LightToAthena()
    my_loader.set_input_uuid('s3://sageworks-incoming-data/aqsol_public_data.csv')
    my_loader.set_output_uuid(output_uuid)

    # Does my S3 data exist?
    assert(my_loader.validate_input())

    # Store this data into Athena/SageWorks
    my_loader.transform()

    # Grab the output and query it
    output = my_loader.get_output()
    query = f"select * from {output_uuid} limit 5"
    df = output.query(query)

    # Setup Pandas output options
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    # Show the dataframe
    print(df)


if __name__ == "__main__":
    test()
