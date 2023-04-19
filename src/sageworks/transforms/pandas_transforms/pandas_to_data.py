"""PandasToData: Class to publish a Pandas DataFrame as a DataSource"""
import awswrangler as wr
import pandas as pd
from pandas.errors import ParserError

# Local imports
from sageworks.utils.sageworks_logging import logging_setup
from sageworks.transforms.transform import Transform, TransformInput, TransformOutput

# Setup Logging
logging_setup()


class PandasToData(Transform):
    """PandasToData: Class to publish a Pandas DataFrame as a DataSource

    Common Usage:
        df_to_data = PandasToData(output_uuid)
        df_to_data.set_output_tags(["test", "small"])
        df_to_data.set_input(test_df)
        df_to_data.transform(delete_existing=True/False)
    """

    def __init__(self, output_uuid: str):
        """PandasToData Initialization"""

        # Call superclass init
        super().__init__("DataFrame", output_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.PANDAS_DF
        self.output_type = TransformOutput.DATA_SOURCE
        self.output_df = None

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
                except (ParserError, ValueError, TypeError):
                    self.log.info(f"Column {c} could not be converted to string...")
        return df

    def transform_impl(self, overwrite: bool = True):
        """Convert the Pandas DataFrame into Parquet Format in the SageWorks S3 Bucket, and
        store the information about the data to the AWS Data Catalog sageworks database"""

        # Set up our metadata storage
        sageworks_meta = {"sageworks_tags": self.output_tags}
        for key, value in self.output_meta.items():
            sageworks_meta[key] = value

        # Create the Output Parquet file S3 Storage Path
        s3_storage_path = f"{self.data_source_s3_path}/{self.output_uuid}"

        # Convert Object Columns to Datetime or String
        self.output_df = self.convert_object_columns(self.output_df)

        # Write out the DataFrame to Parquet/DataStore/Athena
        description = f"SageWorks data source: {self.output_uuid}"
        glue_table_settings = {"description": description, "parameters": sageworks_meta}
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


if __name__ == "__main__":
    """Exercise the PandasToData Class"""
    import sys
    from pathlib import Path

    # Load some small test data
    # Local/relative path to CSV file (FIXME?)
    data_path = Path(sys.modules["sageworks"].__file__).parent.parent.parent / "data" / "test_data.csv"
    test_df = pd.read_csv(data_path)

    # Create my DF to Data Source Transform
    output_uuid = "test_data"
    df_to_data = PandasToData(output_uuid)
    df_to_data.set_input(test_df)
    df_to_data.set_output_tags(["test", "small"])

    # Store this data into a SageWorks DataSource
    df_to_data.transform()
    print(f"{output_uuid} stored as a SageWorks DataSource")
