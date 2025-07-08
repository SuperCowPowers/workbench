"""S3ToDataSourceLight: Class to move LIGHT S3 Files into a Workbench DataSource"""

import sys
import awswrangler as wr

# Local imports
from workbench.core.transforms.transform import Transform, TransformInput, TransformOutput
from workbench.core.transforms.pandas_transforms.pandas_to_data import PandasToData
from workbench.utils.pandas_utils import convert_object_columns


class S3ToDataSourceLight(Transform):
    """S3ToDataSourceLight: Class to move LIGHT S3 Files into a Workbench DataSource

    Common Usage:
        ```python
        s3_to_data = S3ToDataSourceLight(s3_path, data_name, datatype="csv/json")
        s3_to_data.set_output_tags(["abalone", "whatever"])
        s3_to_data.transform()
        ```
    """

    def __init__(self, s3_path: str, data_name: str, datatype: str = "csv"):
        """S3ToDataSourceLight Initialization

        Args:
            s3_path (str): The S3 Path to the file to be transformed
            data_name (str): The Name of the Workbench DataSource to be created
            datatype (str): The datatype of the file to be transformed (defaults to "csv")
        """

        # Call superclass init
        super().__init__(s3_path, data_name)

        # Set up all my instance attributes
        self.input_type = TransformInput.S3_OBJECT
        self.output_type = TransformOutput.DATA_SOURCE
        self.datatype = datatype

    def input_size_mb(self) -> int:
        """Get the size of the input S3 object in MBytes"""
        size_in_bytes = wr.s3.size_objects(self.input_name, boto3_session=self.boto3_session)[self.input_name]
        size_in_mb = round(size_in_bytes / 1_000_000)
        return size_in_mb

    def transform_impl(self, overwrite: bool = True):
        """Convert the S3 CSV data into Parquet Format in the Workbench Data Sources Bucket, and
        store the information about the data to the AWS Data Catalog workbench database
        """

        # Sanity Check for S3 Object size
        object_megabytes = self.input_size_mb()
        if object_megabytes > 100:
            self.log.error(f"S3 Object too big ({object_megabytes} MBytes): Use the S3ToDataSourceHeavy class!")
            return

        # Read in the S3 CSV as a Pandas DataFrame
        if self.datatype == "csv":
            df = wr.s3.read_csv(self.input_name, low_memory=False, boto3_session=self.boto3_session)
        else:
            df = wr.s3.read_json(self.input_name, lines=True, boto3_session=self.boto3_session)

        # Temporary hack to limit the number of columns in the dataframe
        if len(df.columns) > 40:
            self.log.warning(f"{self.input_name} Too Many Columns! Talk to Workbench Support...")

        # Convert object columns before sending to Workbench Data Source
        df = convert_object_columns(df)

        # Use the Workbench Pandas to Data Source class
        pandas_to_data = PandasToData(self.output_name)
        pandas_to_data.set_input(df)
        pandas_to_data.set_output_tags(self.output_tags)
        pandas_to_data.add_output_meta(self.output_meta)
        pandas_to_data.transform()

        # Report the transformation results
        self.log.info(f"{self.input_name} -->  DataSource: {self.output_name} Complete!")

    def post_transform(self, **kwargs):
        """Post-Transform"""
        self.log.info("Post-Transform: S3 to DataSource...")

        # Note: We do not need to onboard because PandasToData already onboarded


if __name__ == "__main__":
    """Exercise the S3ToDataSourceLight Class"""
    from workbench.utils.config_manager import ConfigManager

    # Grab our Workbench Bucket from ENV
    cm = ConfigManager()
    workbench_bucket = cm.get_config("WORKBENCH_BUCKET")
    if workbench_bucket is None:
        print("Could not find ENV var for WORKBENCH_BUCKET!")
        sys.exit(1)

    # Create my Data Loader
    input_path = "s3://" + workbench_bucket + "/incoming-data/aqsol_public_data.csv"
    output_name = "test"
    my_loader = S3ToDataSourceLight(input_path, output_name)
    my_loader.set_output_tags(["test"])

    # Store this data as a Workbench DataSource
    my_loader.transform()
