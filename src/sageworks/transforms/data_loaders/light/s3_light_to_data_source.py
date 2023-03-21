"""S3LightToDataSource: Class to move LIGHT S3 Files into a SageWorks DataSource"""
import awswrangler as wr
import json

# Local imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.artifacts.data_sources.athena_source import AthenaSource


class S3LightToDataSource(Transform):
    def __init__(self, input_uuid=None, output_uuid=None):
        """S3LightToDataSource: Class to move LIGHT S3 Files into a SageWorks DataSource"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.S3_OBJECT
        self.output_type = TransformOutput.DATA_SOURCE

    def input_size_mb(self) -> int:
        """ Get the size of the input S3 object in MBytes"""
        size_in_bytes = wr.s3.size_objects(self.input_uuid, boto3_session=self.boto_session)[self.input_uuid]
        size_in_mb = round(size_in_bytes/1_000_000)
        return size_in_mb

    def transform_impl(self, overwrite: bool = True):
        """Convert the CSV data into Parquet Format in the SageWorks S3 Bucket, and
           store the information about the data to the AWS Data Catalog sageworks database"""

        # Sanity Check for S3 Object size
        object_megabytes = self.input_size_mb()
        if object_megabytes > 100:
            self.log.error(f"S3 Object too big ({object_megabytes} MBytes): Use the S3HeavyToDataSource class!")
            return

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


# Simple test of the S3LightToDataSource functionality
def test():
    """Test the S3LightToDataSource Class"""

    # Create my Data Loader
    output_uuid = 'aqsol_data'
    my_loader = S3LightToDataSource('s3://scp-sageworks-artifacts/incoming-data/aqsol_public_data.csv', output_uuid)

    # Store this data as a SageWorks DataSource
    my_loader.transform()

    # Grab the output and print out it's AWS UUID and tags
    output = AthenaSource(output_uuid)
    print(f"UUID: {output.uuid()}   TAGS: {output.tags()}")

    # Now query the data source and print out the resulting dataframe
    """Commenting out for now"""
    """
    df = output.query(f"select * from {output_uuid} limit 5")

    # Setup Pandas output options
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    # Show the dataframe
    print(df)
    """


if __name__ == "__main__":
    test()
