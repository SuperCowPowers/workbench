"""PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet"""

import pandas as pd
import time
import re
from botocore.exceptions import ClientError
from sagemaker.feature_store.feature_group import FeatureGroup, IngestionError
from sagemaker.feature_store.inputs import TableFormatEnum

# Local imports
from sageworks.utils.iso_8601 import datetime_to_iso8601
from sageworks.core.transforms.transform import Transform, TransformInput, TransformOutput
from sageworks.core.artifacts.artifact import Artifact
from sageworks.core.artifacts.feature_set_core import FeatureSetCore


class PandasToFeatures(Transform):
    """PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet

    Common Usage:
        ```
        to_features = PandasToFeatures(output_uuid)
        to_features.set_output_tags(["abalone", "public", "whatever"])
        to_features.set_input(df, id_column="id"/None, event_time_column="date"/None)
        to_features.transform()
        ```
    """

    def __init__(self, output_uuid: str, auto_one_hot=False):
        """PandasToFeatures Initialization
        Args:
            output_uuid (str): The UUID of the FeatureSet to create
            auto_one_hot (bool): Should we automatically one-hot encode categorical columns?
        """

        # Make sure the output_uuid is a valid UUID
        output_uuid = Artifact.base_compliant_uuid(output_uuid)

        # Call superclass init
        super().__init__("DataFrame", output_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.PANDAS_DF
        self.output_type = TransformOutput.FEATURE_SET
        self.target_column = None
        self.id_column = None
        self.event_time_column = None
        self.auto_one_hot = auto_one_hot
        self.categorical_dtypes = {}
        self.output_df = None
        self.table_format = TableFormatEnum.ICEBERG

        # Delete the existing FeatureSet if it exists
        self.delete_existing()

        # These will be set in the transform method
        self.output_feature_group = None
        self.output_feature_set = None
        self.expected_rows = 0

    def set_input(self, input_df: pd.DataFrame, target_column=None, id_column=None, event_time_column=None):
        """Set the Input DataFrame for this Transform
        Args:
            input_df (pd.DataFrame): The input DataFrame
            target_column (str): The name of the target column (default: None)
            id_column (str): The name of the id column (default: None)
            event_time_column (str): The name of the event_time column (default: None)
        """
        self.target_column = target_column
        self.id_column = id_column
        self.event_time_column = event_time_column
        self.output_df = input_df.copy()

        # Now Prepare the DataFrame for its journey into an AWS FeatureGroup
        self.prep_dataframe()

    def delete_existing(self):
        # Delete the existing FeatureSet if it exists
        try:
            delete_fs = FeatureSetCore(self.output_uuid)
            if delete_fs.exists():
                self.log.info(f"Deleting the {self.output_uuid} FeatureSet...")
                delete_fs.delete()
                time.sleep(1)
        except ClientError as exc:
            self.log.info(f"FeatureSet {self.output_uuid} doesn't exist...")
            self.log.info(exc)

    def _ensure_id_column(self):
        """Internal: AWS Feature Store requires an Id field for all data store"""
        if self.id_column is None or self.id_column not in self.output_df.columns:
            if "id" not in self.output_df.columns:
                self.log.info("Generating an id column before FeatureSet Creation...")
                self.output_df["id"] = self.output_df.index
            self.id_column = "id"

    def _ensure_event_time(self):
        """Internal: AWS Feature Store requires an event_time field for all data stored"""
        if self.event_time_column is None or self.event_time_column not in self.output_df.columns:
            self.log.info("Generating an event_time column before FeatureSet Creation...")
            self.event_time_column = "event_time"
            self.output_df[self.event_time_column] = pd.Timestamp("now", tz="UTC")

        # The event_time_column is defined, so we need to make sure it's in ISO-8601 string format
        # Note: AWS Feature Store only a particular ISO-8601 format not ALL ISO-8601 formats
        time_column = self.output_df[self.event_time_column]

        # Check if the event_time_column is of type object or string convert it to DateTime
        if time_column.dtypes == "object" or time_column.dtypes.name == "string":
            self.log.info(f"Converting {self.event_time_column} to DateTime...")
            time_column = pd.to_datetime(time_column)

        # Let's make sure it the right type for Feature Store
        if pd.api.types.is_datetime64_any_dtype(time_column):
            self.log.info(f"Converting {self.event_time_column} to ISOFormat Date String before FeatureSet Creation...")

            # Convert the datetime DType to ISO-8601 string
            # TableFormat=ICEBERG does not support alternate formats for event_time field, it only supports String type.
            time_column = time_column.map(datetime_to_iso8601)
            self.output_df[self.event_time_column] = time_column.astype("string")

    def _convert_objs_to_string(self):
        """Internal: AWS Feature Store doesn't know how to store object dtypes, so convert to String"""
        for col in self.output_df:
            if pd.api.types.is_object_dtype(self.output_df[col].dtype):
                self.output_df[col] = self.output_df[col].astype(pd.StringDtype())

    def process_column_name(self, column: str, shorten: bool = False) -> str:
        """Call various methods to make sure the column is ready for Feature Store
        Args:
            column (str): The column name to process
            shorten (bool): Should we shorten the column name? (default: False)
        """
        self.log.debug(f"Processing column {column}...")

        # Make sure the column name is valid
        column = self.sanitize_column_name(column)

        # Make sure the column name isn't too long
        if shorten:
            column = self.shorten_column_name(column)

        return column

    def shorten_column_name(self, name, max_length=20):
        if len(name) <= max_length:
            return name

        # Start building the new name from the end
        parts = name.split("_")[::-1]
        new_name = ""
        for part in parts:
            if len(new_name) + len(part) + 1 <= max_length:  # +1 for the underscore
                new_name = f"{part}_{new_name}" if new_name else part
            else:
                break

        # If new_name is empty, just use the last part of the original name
        if not new_name:
            new_name = parts[0]

        self.log.info(f"Shortening {name} to {new_name}")
        return new_name

    def sanitize_column_name(self, name):
        # Remove all invalid characters
        sanitized = re.sub("[^a-zA-Z0-9-_]", "_", name)
        sanitized = re.sub("_+", "_", sanitized)
        sanitized = sanitized.strip("_")

        # Log the change if the name was altered
        if sanitized != name:
            self.log.info(f"Sanitizing {name} to {sanitized}")

        return sanitized

    def one_hot_encoding(self, df, categorical_columns: list) -> pd.DataFrame:
        """One Hot Encoding for Categorical Columns with additional column name management"""

        # Now convert Categorical Types to One Hot Encoding
        current_columns = list(df.columns)
        df = pd.get_dummies(df, columns=categorical_columns)

        # Compute the new columns generated by get_dummies
        new_columns = list(set(df.columns) - set(current_columns))

        # Convert new columns to int32
        df[new_columns] = df[new_columns].astype("int32")

        # For the new columns we're going to shorten the names
        renamed_columns = {col: self.process_column_name(col) for col in new_columns}

        # Rename the columns in the DataFrame
        df.rename(columns=renamed_columns, inplace=True)

        return df

    # Helper Methods
    def auto_convert_columns_to_categorical(self):
        """Convert object and string types to Categorical"""
        categorical_columns = []
        for feature, dtype in self.output_df.dtypes.items():
            if dtype in ["object", "string", "category"] and feature not in [
                self.event_time_column,
                self.id_column,
                self.target_column,
            ]:
                unique_values = self.output_df[feature].nunique()
                if 1 < unique_values < 6:
                    self.log.important(f"Converting column {feature} to categorical (unique {unique_values})")
                    self.output_df[feature] = self.output_df[feature].astype("category")
                    categorical_columns.append(feature)

        # Now run one hot encoding on categorical columns
        self.output_df = self.one_hot_encoding(self.output_df, categorical_columns)

    def manual_categorical_converter(self):
        """Convert object and string types to Categorical

        Note:
            This method is used for streaming/chunking. You can set the
            categorical_dtypes attribute to a dictionary of column names and
            their respective categorical types.
        """
        for column, cat_d_type in self.categorical_dtypes.items():
            self.output_df[column] = self.output_df[column].astype(cat_d_type)

        # Now convert Categorical Types to One Hot Encoding
        categorical_columns = list(self.categorical_dtypes.keys())
        self.output_df = self.one_hot_encoding(self.output_df, categorical_columns)

    @staticmethod
    def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
        """Convert the types of the DataFrame to the correct types for the Feature Store"""
        for column in list(df.select_dtypes(include="bool").columns):
            df[column] = df[column].astype("int32")
        for column in list(df.select_dtypes(include="category").columns):
            df[column] = df[column].astype("str")

        # Special case for datetime types
        for column in df.select_dtypes(include=["datetime"]).columns:
            df[column] = df[column].map(datetime_to_iso8601).astype("string")

        """FIXME Not sure we need these conversions
        for column in list(df.select_dtypes(include="object").columns):
            df[column] = df[column].astype("string")
        for column in list(df.select_dtypes(include=[pd.Int64Dtype]).columns):
            df[column] = df[column].astype("int64")
        for column in list(df.select_dtypes(include=[pd.Float64Dtype]).columns):
            df[column] = df[column].astype("float64")
        """
        return df

    def prep_dataframe(self):
        """Prep the DataFrame for Feature Store Creation"""
        self.log.info("Prep the output_df (cat_convert, convert types, and lowercase columns)...")

        # Make sure we have the required id and event_time columns
        self._ensure_id_column()
        self._ensure_event_time()

        # Convert object and string types to Categorical
        if self.auto_one_hot:
            self.auto_convert_columns_to_categorical()
        else:
            self.manual_categorical_converter()

        # We need to convert some of our column types to the correct types
        # Feature Store only supports these data types:
        # - Integral
        # - Fractional
        # - String (timestamp/datetime types need to be converted to string)
        self.output_df = self.convert_column_types(self.output_df)

        # Convert columns names to lowercase, Athena will not work with uppercase column names
        if str(self.output_df.columns) != str(self.output_df.columns.str.lower()):
            for c in self.output_df.columns:
                if c != c.lower():
                    self.log.important(f"Column name {c} converted to lowercase: {c.lower()}")
            self.output_df.columns = self.output_df.columns.str.lower()

    def create_feature_group(self):
        """Create a Feature Group, load our Feature Definitions, and wait for it to be ready"""

        # Create a Feature Group and load our Feature Definitions
        my_feature_group = FeatureGroup(name=self.output_uuid, sagemaker_session=self.sm_session)
        my_feature_group.load_feature_definitions(data_frame=self.output_df)

        # Create the Output S3 Storage Path for this Feature Set
        s3_storage_path = f"{self.feature_sets_s3_path}/{self.output_uuid}"

        # Get the metadata/tags to push into AWS
        aws_tags = self.get_aws_tags()

        # Create the Feature Group
        my_feature_group.create(
            s3_uri=s3_storage_path,
            record_identifier_name=self.id_column,
            event_time_feature_name=self.event_time_column,
            role_arn=self.sageworks_role_arn,
            enable_online_store=True,
            table_format=self.table_format,
            tags=aws_tags,
        )

        # Ensure/wait for the feature group to be created
        self.ensure_feature_group_created(my_feature_group)
        return my_feature_group

    def pre_transform(self, **kwargs):
        """Pre-Transform: Create the Feature Group"""
        self.output_feature_group = self.create_feature_group()

    def transform_impl(self):
        """Transform Implementation: Ingest the data into the Feature Group"""

        # Now we actually push the data into the Feature Group (called ingestion)
        self.log.important("Ingesting rows into Feature Group...")
        ingest_manager = self.output_feature_group.ingest(self.output_df, max_processes=8, wait=False)
        try:
            ingest_manager.wait()
        except IngestionError as exc:
            self.log.warning(f"Some rows had an ingesting error: {exc}")

        # Report on any rows that failed to ingest
        if ingest_manager.failed_rows:
            self.log.warning(f"Number of Failed Rows: {len(ingest_manager.failed_rows)}")

            # FIXME: This may or may not give us the correct rows
            # If any index is greater then the number of rows, then the index needs
            # to be converted to a relative index in our current output_df
            df_rows = len(self.output_df)
            relative_indexes = [idx - df_rows if idx >= df_rows else idx for idx in ingest_manager.failed_rows]
            failed_data = self.output_df.iloc[relative_indexes]
            for idx, row in failed_data.iterrows():
                self.log.warning(f"Failed Row {idx}: {row.to_dict()}")

        # Keep track of the number of rows we expect to be ingested
        self.expected_rows += len(self.output_df) - len(ingest_manager.failed_rows)
        self.log.info(f"Added rows: {len(self.output_df)}")
        self.log.info(f"Failed rows: {len(ingest_manager.failed_rows)}")
        self.log.info(f"Total rows to be ingested: {self.expected_rows}")

    def post_transform(self, **kwargs):
        """Post-Transform: Populating Offline Storage and onboard()"""
        self.log.info("Post-Transform: Populating Offline Storage and onboard()...")

        # Feature Group Ingestion takes a while, so we need to wait for it to finish
        self.output_feature_set = FeatureSetCore(self.output_uuid, force_refresh=True)
        self.log.important("Waiting for AWS Feature Group Offline storage to be ready...")
        self.log.important("This will often take 10-20 minutes...go have coffee or lunch :)")
        self.output_feature_set.set_status("initializing")
        self.wait_for_rows(self.expected_rows)

        # Call the FeatureSet onboard method to compute a bunch of EDA stuff
        self.output_feature_set.onboard()

    def ensure_feature_group_created(self, feature_group):
        status = feature_group.describe().get("FeatureGroupStatus")
        while status == "Creating":
            self.log.debug("FeatureSet being Created...")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
        self.log.info(f"FeatureSet {feature_group.name} successfully created")

    def wait_for_rows(self, expected_rows: int):
        """Wait for AWS Feature Group to fully populate the Offline Storage"""
        rows = self.output_feature_set.num_rows()

        # Wait for the rows to be populated
        self.log.info(f"Waiting for AWS Feature Group {self.output_uuid} Offline Storage...")
        not_all_rows_retry = 5
        while rows < expected_rows and not_all_rows_retry > 0:
            sleep_time = 5 if rows else 60
            not_all_rows_retry -= 1 if rows else 0
            time.sleep(sleep_time)
            rows = self.output_feature_set.num_rows()
            self.log.info(f"Offline Storage {self.output_uuid}: {rows} rows out of {expected_rows}")
        if rows == expected_rows:
            self.log.important(f"Success: Reached Expected Rows ({rows} rows)...")
        else:
            self.log.warning(
                f"Did not reach expected rows ({rows}/{expected_rows}) but we're not sweating the small stuff..."
            )


if __name__ == "__main__":
    """Exercise the PandasToFeatures Class"""
    from sageworks.api.data_source import DataSource

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 15)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Crab the test_data DataSource
    ds = DataSource("test_data")
    data_df = ds.sample()

    # Create my DF to Feature Set Transform
    df_to_features = PandasToFeatures("test_features", auto_one_hot=True)
    df_to_features.set_input(data_df, id_column="id")
    df_to_features.set_output_tags(["test", "small"])

    # Store this dataframe as a SageWorks Feature Set
    df_to_features.transform()

    # Test non-compliant output UUID
    df_to_features = PandasToFeatures("test_features-123")
