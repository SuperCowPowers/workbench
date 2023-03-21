"""S3HeavyToDataSource: Class to move HEAVY S3 Files into a SageWorks DataSource"""
import awswrangler as wr
import json

# Local imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.artifacts.data_sources.athena_source import AthenaSource


# FIXME: This class is not funny implemented. It will be a glue jobs that uses SPARK to convert
#        CSV to Parquet in a scalable way and then populate the AWS Data Catalog with the information


class S3HeavyToDataSource(Transform):
    def __init__(self, input_uuid=None, output_uuid=None):
        """S3HeavyToDataSource: Class to move HEAVY S3 Files into a SageWorks DataSource"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.S3_OBJECT
        self.output_type = TransformOutput.DATA_SOURCE

    def transform_impl(self, overwrite: bool = True):
        """Convert the CSV data into Parquet Format in the SageWorks S3 Bucket, and
           store the information about the data to the AWS Data Catalog sageworks database"""

        # Add some tags here
        tags = ['sageworks', 'public']

        # Read in the S3 CSV as a Pandas DataFrame
        df = wr.s3.read_csv(self.input_uuid, low_memory=False, boto3_session=self.boto_session)

        # Create the Output Parquet file S3 Storage Path
        s3_storage_path = f"{self.data_source_s3_path}/{self.output_uuid}"

        # Write out the DataFrame to Parquet/DataStore/Athena
        wr.s3.to_parquet(df, path=s3_storage_path, dataset=True, mode='overwrite',
                         database=self.data_catalog_db, table=self.output_uuid,
                         description=f'SageWorks data source: {self.output_uuid}',
                         filename_prefix=f'{self.output_uuid}_',
                         parameters={'tags': json.dumps(tags)},
                         boto3_session=self.boto_session,
                         partition_cols=None)  # FIXME: Have some logic around partition columns


# Simple test of the S3HeavyToDataSource functionality
def test():
    """Test the S3HeavyToDataSource Class"""
    import pandas as pd

    # Create my Data Loader
    output_uuid = 'aqsol_data'
    my_loader = S3HeavyToDataSource('s3://scp-sageworks-artifacts/incoming-data/aqsol_public_data.csv', output_uuid)

    # Store this data as a SageWorks DataSource
    my_loader.transform()

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
