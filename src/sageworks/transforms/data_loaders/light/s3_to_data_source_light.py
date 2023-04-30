"""S3ToDataSourceLight: Class to move LIGHT S3 Files into a SageWorks DataSource"""
import awswrangler as wr
import pandas as pd
from pandas.errors import ParserError

# Local imports
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.transforms.pandas_transforms.pandas_to_data import PandasToData


class S3ToDataSourceLight(Transform):
    """S3ToDataSourceLight: Class to move LIGHT S3 Files into a SageWorks DataSource

    Common Usage:
        s3_to_data = S3ToDataSourceLight(s3_path, data_uuid, datatype="csv/json")
        s3_to_data.set_output_tags(["abalone", "whatever"])
        s3_to_data.transform()
    """

    def __init__(self, s3_path: str, data_uuid: str, datatype: str = "csv"):
        """S3ToDataSourceLight Initialization
        Args:
            s3_path (str): The S3 Path to the file to be transformed
            data_uuid (str): The UUID of the SageWorks DataSource to be created
            datatype (str): The datatype of the file to be transformed (defaults to "csv")
        """

        # Call superclass init
        super().__init__(s3_path, data_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.S3_OBJECT
        self.output_type = TransformOutput.DATA_SOURCE
        self.datatype = datatype

    def input_size_mb(self) -> int:
        """Get the size of the input S3 object in MBytes"""
        size_in_bytes = wr.s3.size_objects(self.input_uuid, boto3_session=self.boto_session)[self.input_uuid]
        size_in_mb = round(size_in_bytes / 1_000_000)
        return size_in_mb

    def convert_object_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Try to automatically convert object columns to datetime columns"""
        for c in df.columns[df.dtypes == "object"]:  # Look at the object columns
            try:
                df[c] = pd.to_datetime(df[c])
            except (ParserError, ValueError, TypeError):
                self.log.debug(f"Column {c} could not be converted to datetime...")

                # Now try to convert object to string
                try:
                    df[c] = df[c].astype(str)
                except (ParserError, ValueError, TypeError):
                    self.log.info(f"Column {c} could not be converted to string...")
        return df

    def transform_impl(self, overwrite: bool = True):
        """Convert the CSV data into Parquet Format in the SageWorks Data Sources Bucket, and
        store the information about the data to the AWS Data Catalog sageworks database"""

        # Sanity Check for S3 Object size
        object_megabytes = self.input_size_mb()
        if object_megabytes > 100:
            self.log.error(f"S3 Object too big ({object_megabytes} MBytes): Use the S3ToDataSourceHeavy class!")
            return

        # Read in the S3 CSV as a Pandas DataFrame
        if self.datatype == "csv":
            df = wr.s3.read_csv(self.input_uuid, low_memory=False, boto3_session=self.boto_session)
        else:
            df = wr.s3.read_json(self.input_uuid, lines=True, boto3_session=self.boto_session)

        # Convert object columns before sending to SageWorks Data Source
        df = self.convert_object_columns(df)

        # Use the SageWorks Pandas to Data Source class
        pandas_to_data = PandasToData(self.output_uuid)
        pandas_to_data.set_input(df)
        pandas_to_data.set_output_tags(self.output_tags)
        pandas_to_data.add_output_meta(self.output_meta)
        pandas_to_data.transform()


if __name__ == "__main__":
    """Exercise the S3ToDataSourceLight Class"""
    from sageworks.utils.sageworks_config import SageWorksConfig

    # Create my Data Loader
    sageworks_config = SageWorksConfig()
    sageworks_bucket = sageworks_config.get_config_value("SAGEWORKS_AWS", "S3_BUCKET_NAME")
    input_path = "s3://" + sageworks_bucket + "/incoming-data/aqsol_public_data.csv"
    output_uuid = "aqsol_data"
    my_loader = S3ToDataSourceLight(input_path, output_uuid)
    my_loader.set_output_tags(["aqsol", "public"])

    # Store this data as a SageWorks DataSource
    my_loader.transform()
