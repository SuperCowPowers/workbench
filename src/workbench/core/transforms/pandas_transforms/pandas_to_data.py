"""PandasToData: Class to publish a Pandas DataFrame as a DataSource"""

import awswrangler as wr
import pandas as pd
from pandas.errors import ParserError
import time

# Local imports
from workbench.utils.datetime_utils import datetime_to_iso8601
from workbench.core.transforms.transform import Transform, TransformInput, TransformOutput
from workbench.core.artifacts.data_source_factory import DataSourceFactory
from workbench.core.artifacts.artifact import Artifact
from workbench.core.artifacts.athena_source import AthenaSource


class PandasToData(Transform):
    """PandasToData: Class to publish a Pandas DataFrame as a DataSource

    Common Usage:
        ```python
        df_to_data = PandasToData(output_uuid)
        df_to_data.set_output_tags(["test", "small"])
        df_to_data.set_input(test_df)
        df_to_data.transform()
        ```
    """

    def __init__(self, output_uuid: str, output_format: str = "parquet", catalog_db: str = "workbench"):
        """PandasToData Initialization
        Args:
            output_uuid (str): The UUID of the DataSource to create
            output_format (str): The file format to store the S3 object data in (default: "parquet")
            catalog_db (str): The AWS Data Catalog Database to use (default: "workbench")
        """

        # Make sure the output_uuid is a valid name/id
        Artifact.is_name_valid(output_uuid)

        # Call superclass init
        super().__init__("DataFrame", output_uuid, catalog_db)

        # Set up all my instance attributes
        self.input_type = TransformInput.PANDAS_DF
        self.output_type = TransformOutput.DATA_SOURCE
        self.output_df = None

        # Give a message that Parquet is best in most cases
        if output_format != "parquet":
            self.log.warning("Parquet format works the best in most cases please consider using it")
        self.output_format = output_format

    def set_input(self, input_df: pd.DataFrame):
        """Set the DataFrame Input for this Transform"""
        self.output_df = input_df.copy()

    def delete_existing(self):
        # Delete the existing FeatureSet if it exists
        self.log.info(f"Deleting the {self.output_uuid} DataSource...")
        AthenaSource.managed_delete(self.output_uuid)
        time.sleep(1)

    def convert_object_to_string(self, df: pd.DataFrame) -> pd.DataFrame:
        """Try to automatically convert object columns to string columns"""
        for c in df.columns[df.dtypes == "object"]:  # Look at the object columns
            try:
                df[c] = df[c].astype("string")
                df[c] = df[c].str.replace("'", '"')  # This is for nested JSON
            except (ParserError, ValueError, TypeError):
                self.log.info(f"Column {c} could not be converted to string...")
        return df

    def convert_object_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Try to automatically convert object columns to datetime or string columns"""
        for c in df.columns[df.dtypes == "object"]:  # Look at the object columns
            try:
                df[c] = pd.to_datetime(df[c])
            except (ParserError, ValueError, TypeError):
                self.log.debug(f"Column {c} could not be converted to datetime...")
        return df

    @staticmethod
    def convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime columns to ISO-8601 string"""
        datetime_type = ["datetime", "datetime64", "datetime64[ns]", "datetimetz"]
        for c in df.select_dtypes(include=datetime_type).columns:
            df[c] = df[c].map(datetime_to_iso8601)
            df[c] = df[c].astype(pd.StringDtype())
        return df

    def pre_transform(self, **kwargs):
        """Pre-Transform: Delete the existing DataSource if it exists"""
        self.delete_existing()

    def transform_impl(self, overwrite: bool = True):
        """Convert the Pandas DataFrame into Parquet Format in the Workbench S3 Bucket, and
        store the information about the data to the AWS Data Catalog workbench database

        Args:
            overwrite (bool): Overwrite the existing data in the Workbench S3 Bucket (default: True)
        """
        self.log.info(f"DataFrame to Workbench DataSource: {self.output_uuid}...")

        # Set up our metadata storage
        workbench_meta = {"workbench_tags": self.output_tags}
        workbench_meta.update(self.output_meta)

        # Create the Output Parquet file S3 Storage Path
        s3_storage_path = f"{self.data_sources_s3_path}/{self.output_uuid}"

        # Convert columns names to lowercase, Athena will not work with uppercase column names
        if str(self.output_df.columns) != str(self.output_df.columns.str.lower()):
            for c in self.output_df.columns:
                if c != c.lower():
                    self.log.important(f"Column name {c} converted to lowercase: {c.lower()}")
            self.output_df.columns = self.output_df.columns.str.lower()

        # Convert Object Columns to String
        self.output_df = self.convert_object_to_string(self.output_df)

        # Note: Both of these conversions may not be necessary, so we're leaving them commented out
        """
        # Convert Object Columns to Datetime
        self.output_df = self.convert_object_to_datetime(self.output_df)

        # Now convert datetime columns to ISO-8601 string
        # self.output_df = self.convert_datetime_columns(self.output_df)
        """

        # Write out the DataFrame to AWS Data Catalog in either Parquet or JSONL format
        description = f"Workbench data source: {self.output_uuid}"
        glue_table_settings = {"description": description, "parameters": workbench_meta}
        if self.output_format == "parquet":
            wr.s3.to_parquet(
                self.output_df,
                path=s3_storage_path,
                dataset=True,
                mode="overwrite",
                database=self.data_catalog_db,
                table=self.output_uuid,
                filename_prefix=f"{self.output_uuid}_",
                boto3_session=self.boto3_session,
                partition_cols=None,
                glue_table_settings=glue_table_settings,
                sanitize_columns=False,
            )  # FIXME: Have some logic around partition columns

        # Note: In general Parquet works will for most uses cases. We recommend using Parquet
        #       You can use JSON_EXTRACT on Parquet string field, and it works great.
        elif self.output_format == "jsonl":
            self.log.warning("We recommend using Parquet format for most use cases")
            self.log.warning("If you have a use case that requires JSONL please contact Workbench support")
            self.log.warning("We'd like to understand what functionality JSONL is providing that isn't already")
            self.log.warning("provided with Parquet and JSON_EXTRACT() for your Athena Queries")
            wr.s3.to_json(
                self.output_df,
                path=s3_storage_path,
                orient="records",
                lines=True,
                date_format="iso",
                dataset=True,
                mode="overwrite",
                database=self.data_catalog_db,
                table=self.output_uuid,
                filename_prefix=f"{self.output_uuid}_",
                boto3_session=self.boto3_session,
                partition_cols=None,
                glue_table_settings=glue_table_settings,
            )
        else:
            raise ValueError(f"Unsupported file format: {self.output_format}")

    def post_transform(self, **kwargs):
        """Post-Transform: Calling onboard() fnr the DataSource"""
        self.log.info("Post-Transform: Calling onboard() for the DataSource...")

        # Onboard the DataSource
        output_data_source = DataSourceFactory(self.output_uuid)
        output_data_source.onboard()


if __name__ == "__main__":
    """Exercise the PandasToData Class"""
    from workbench.utils.test_data_generator import TestDataGenerator

    # Generate some test data
    test_data = TestDataGenerator()
    df = test_data.person_data()

    # Create my Pandas to DataSource Transform
    test_uuid = "test_data"
    df_to_data = PandasToData(test_uuid)
    df_to_data.set_input(df)
    df_to_data.set_output_tags(["test", "small"])
    df_to_data.transform()
    print(f"{test_uuid} stored as a Workbench DataSource")

    # Test column names with uppercase
    df.rename(columns={"iq_score": "IQ_Score"}, inplace=True)
    df_to_data.set_input(df)
    df_to_data.transform()

    # Create my Pandas to DataSource using a JSONL format
    """
    data_path = Path(sys.modules["workbench"].__file__).parent.parent.parent / "data" / "test_data.json"
    test_df = pd.read_json(data_path, orient="records", lines=True)
    output_uuid = "test_data_json"
    df_to_data = PandasToData(output_uuid)
    df_to_data.set_input(test_df)
    df_to_data.set_output_tags(["test", "json"])

    # Store this data into a Workbench DataSource
    df_to_data.transform()
    print(f"{output_uuid} stored as a Workbench DataSource")
    """
