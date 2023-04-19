"""PandasToData: Class to publish a Pandas DataFrame as a DataSource"""
import awswrangler as wr
import pandas as pd
from pandas.errors import ParserError

# Local imports
from sageworks.utils.iso_8601 import datetime_to_iso8601
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput


class PandasToData(Transform):
    """PandasToData: Class to publish a Pandas DataFrame as a DataSource

    Common Usage:
        df_to_data = PandasToData(output_uuid)
        df_to_data.set_output_tags(["test", "small"])
        df_to_data.set_input(test_df)
        df_to_data.transform(delete_existing=True/False)
    """

    def __init__(self, output_uuid: str, output_file_format: str = "parquet", **kwargs):
        """PandasToData Initialization
        Args:
            output_uuid (str): The UUID of the DataSource to create
            output_file_format (str): The file format to store the S3 object data in
        """

        # Call superclass init
        super().__init__("DataFrame", output_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.PANDAS_DF
        self.output_type = TransformOutput.DATA_SOURCE
        self.output_df = None

        # Give a message that Parquet is best in most cases
        if output_file_format != "parquet":
            self.log.warning("Parquet format works the best in most cases please consider using it")
        self.output_file_format = output_file_format

    def set_input(self, input_df: pd.DataFrame):
        """Set the DataFrame Input for this Transform"""
        self.output_df = input_df.copy()

    def convert_object_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Try to automatically convert object columns to datetime or string columns"""
        for c in df.columns[df.dtypes == "object"]:  # Look at the object columns
            try:
                df[c] = pd.to_datetime(df[c])
            except (ParserError, ValueError, TypeError):
                self.log.debug(f"Column {c} could not be converted to datetime...")

                # Now try to convert object to string
                try:
                    df[c] = df[c].astype(str)
                    df[c] = df[c].str.replace("'", '"')  # This is for nested JSON
                except (ParserError, ValueError, TypeError):
                    self.log.info(f"Column {c} could not be converted to string...")
        return df

    @staticmethod
    def convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime columns to ISO-8601 string"""
        datetime_type = ["datetime", "datetime64", "datetime64[ns]", "datetimetz"]
        for c in df.select_dtypes(include=datetime_type).columns:
            df[c] = df[c].map(datetime_to_iso8601)
            df[c] = df[c].astype(pd.StringDtype())
        return df

    def transform_impl(self, overwrite: bool = True):
        """Convert the Pandas DataFrame into Parquet Format in the SageWorks S3 Bucket, and
        store the information about the data to the AWS Data Catalog sageworks database

        Args:
            overwrite (bool): Overwrite the existing data in the SageWorks S3 Bucket
        """

        # Set up our metadata storage
        sageworks_meta = {"sageworks_tags": self.output_tags}
        for key, value in self.output_meta.items():
            sageworks_meta[key] = value

        # Create the Output Parquet file S3 Storage Path
        s3_storage_path = f"{self.data_source_s3_path}/{self.output_uuid}"

        # Convert Object Columns to Datetime or String
        self.output_df = self.convert_object_columns(self.output_df)

        # Now convert datetime columns to ISO-8601 string
        self.output_df = self.convert_datetime_columns(self.output_df)

        # Write out the DataFrame to AWS Data Catalog in either Parquet or JSONL format
        description = f"SageWorks data source: {self.output_uuid}"
        glue_table_settings = {"description": description, "parameters": sageworks_meta}
        if self.output_file_format == "parquet":
            wr.s3.to_parquet(
                self.output_df,
                path=s3_storage_path,
                dataset=True,
                mode="overwrite",
                database=self.data_catalog_db,
                table=self.output_uuid,
                filename_prefix=f"{self.output_uuid}_",
                boto3_session=self.boto_session,
                partition_cols=None,
                glue_table_settings=glue_table_settings,
            )  # FIXME: Have some logic around partition columns

        # Note: In general Parquet works will for most uses cases. We recommend using Parquet
        #       You can use JSON_EXTRACT on Parquet string field, and it works great.
        elif self.output_file_format == "jsonl":
            self.log.warning("We recommend using Parquet format for most use cases")
            self.log.warning("If you have a use case that requires JSONL please contact SageWorks support")
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
                boto3_session=self.boto_session,
                partition_cols=None,
                glue_table_settings=glue_table_settings,
            )
        else:
            raise ValueError(f"Unsupported file format: {self.output_file_format}")


if __name__ == "__main__":
    """Exercise the PandasToData Class"""
    import sys
    from pathlib import Path

    # Load some small test data
    # Local/relative path to CSV file (FIXME?)
    data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "test_data.csv"
    test_df = pd.read_csv(data_path)

    # Create my DF to Data Source Transform
    my_data_name = "test_data"
    df_to_data = PandasToData(my_data_name)
    df_to_data.set_input(test_df)
    df_to_data.set_output_tags(["test", "small"])

    # Store this data into a SageWorks DataSource
    df_to_data.transform()
    print(f"{my_data_name} stored as a SageWorks DataSource")

    # Create my DF to with JSONL format
    data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "test_data.json"
    test_df = pd.read_json(data_path, orient="records", lines=True)
    output_uuid = "test_data_json"
    df_to_data = PandasToData(output_uuid)
    df_to_data.set_input(test_df)
    df_to_data.set_output_tags(["test", "json"])

    # Store this data into a SageWorks DataSource
    df_to_data.transform()
    print(f"{output_uuid} stored as a SageWorks DataSource")
