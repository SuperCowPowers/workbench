"""S3ToDataSourceLight: Class to move LIGHT S3 Files into a SageWorks DataSource"""
import awswrangler as wr

# Local imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.transforms.pandas_transforms.pandas_to_data import PandasToData


class S3ToDataSourceLight(Transform):
    def __init__(self, input_uuid=None, output_uuid=None):
        """S3ToDataSourceLight: Class to move LIGHT S3 Files into a SageWorks DataSource"""

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
        """Convert the CSV data into Parquet Format in the SageWorks Data Sources Bucket, and
           store the information about the data to the AWS Data Catalog sageworks database"""

        # Sanity Check for S3 Object size
        object_megabytes = self.input_size_mb()
        if object_megabytes > 100:
            self.log.error(f"S3 Object too big ({object_megabytes} MBytes): Use the S3ToDataSourceHeavy class!")
            return

        # Read in the S3 CSV as a Pandas DataFrame
        df = wr.s3.read_csv(self.input_uuid, low_memory=False, boto3_session=self.boto_session)

        # Use the SageWorks Pandas to Data Source class
        pandas_to_data = PandasToData(self.output_uuid)
        pandas_to_data.set_input(df)
        pandas_to_data.transform()


# Simple test of the S3ToDataSourceLight functionality
def test():
    """Test the S3ToDataSourceLight Class"""

    # Create my Data Loader
    output_uuid = 'aqsol_data'
    my_loader = S3ToDataSourceLight('s3://scp-sageworks-artifacts/incoming-data/aqsol_public_data.csv', output_uuid)

    # Store this data as a SageWorks DataSource
    my_loader.transform()


if __name__ == "__main__":
    test()
