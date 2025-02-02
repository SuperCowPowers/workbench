"""PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet"""

import pandas as pd
import time
import re
from sagemaker.feature_store.feature_group import FeatureGroup, IngestionError
from sagemaker.feature_store.inputs import TableFormatEnum

# Local imports
from workbench.utils.datetime_utils import datetime_to_iso8601
from workbench.core.transforms.transform import Transform, TransformInput, TransformOutput
from workbench.core.artifacts.artifact import Artifact
from workbench.core.artifacts.feature_set_core import FeatureSetCore


class PandasToFeatures(Transform):
    """PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet

    Common Usage:
        ```python
        to_features = PandasToFeatures(output_uuid)
        to_features.set_output_tags(["my", "awesome", "data"])
        to_features.set_input(df, id_column="my_id")
        to_features.transform()
        ```
    """

    def __init__(self, output_uuid: str):
        """PandasToFeatures Initialization

        Args:
            output_uuid (str): The UUID of the FeatureSet to create
        """

        # Make sure the output_uuid is a valid name
        Artifact.is_name_valid(output_uuid)

        # Call superclass init
        super().__init__("DataFrame", output_uuid)

        # Set up all my instance attributes
        self.input_type = TransformInput.PANDAS_DF
        self.output_type = TransformOutput.FEATURE_SET
        self.id_column = None
        self.event_time_column = None
        self.one_hot_columns = []
        self.categorical_dtypes = {}  # Used for streaming/chunking
        self.output_df = None
        self.table_format = TableFormatEnum.ICEBERG
        self.incoming_hold_out_ids = None

        # These will be set in the transform method
        self.output_feature_group = None
        self.output_feature_set = None
        self.expected_rows = 0

    def set_input(self, input_df: pd.DataFrame, id_column=None, event_time_column=None, one_hot_columns=None):
        """Set the Input DataFrame for this Transform

        Args:
            input_df (pd.DataFrame): The input DataFrame.
            id_column (str, optional): The ID column (if not specified, an 'auto_id' will be generated).
            event_time_column (str, optional): The name of the event time column (default: None).
            one_hot_columns (list, optional): The list of columns to one-hot encode (default: None).
        """
        self.id_column = id_column
        self.event_time_column = event_time_column
        self.output_df = input_df.copy()
        self.one_hot_columns = one_hot_columns or []

        # Now Prepare the DataFrame for its journey into an AWS FeatureGroup
        self.prep_dataframe()

    def delete_existing(self):
        # Delete the existing FeatureSet if it exists
        self.log.info(f"Deleting the {self.output_uuid} FeatureSet...")
        FeatureSetCore.managed_delete(self.output_uuid)
        time.sleep(1)

    def _ensure_id_column(self):
        """Internal: AWS Feature Store requires an Id field"""
        if self.id_column is None:
            self.log.warning("Generating an 'auto_id' column, we highly recommended setting the 'id_column'")
            self.output_df["auto_id"] = self.output_df.index
            self.id_column = "auto_id"
            return
        if self.id_column not in self.output_df.columns:
            error_msg = f"Id column {self.id_column} not found in the DataFrame"
            self.log.critical(error_msg)
            raise ValueError(error_msg)

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

    def one_hot_encode(self, df, one_hot_columns: list) -> pd.DataFrame:
        """One Hot Encoding for Categorical Columns with additional column name management

        Args:
            df (pd.DataFrame): The DataFrame to process
            one_hot_columns (list): The list of columns to one-hot encode

        Returns:
            pd.DataFrame: The DataFrame with one-hot encoded columns
        """

        # Grab the current list of columns
        current_columns = list(df.columns)

        # Now convert the list of columns into Categorical and then One-Hot Encode
        self.convert_columns_to_categorical(one_hot_columns)
        self.log.important(f"One-Hot encoding columns: {one_hot_columns}")
        df = pd.get_dummies(df, columns=one_hot_columns)

        # Compute the new columns generated by get_dummies
        new_columns = list(set(df.columns) - set(current_columns))
        self.log.important(f"New columns generated: {new_columns}")

        # Convert new columns to int32
        df[new_columns] = df[new_columns].astype("int32")

        # For the new columns we're going to shorten the names
        renamed_columns = {col: self.process_column_name(col) for col in new_columns}

        # Rename the columns in the DataFrame
        df.rename(columns=renamed_columns, inplace=True)

        return df

    # Helper Methods
    def convert_columns_to_categorical(self, columns: list):
        """Convert column to Categorical type"""
        for feature in columns:
            if feature not in [self.event_time_column, self.id_column]:
                unique_values = self.output_df[feature].nunique()
                if 1 < unique_values < 10:
                    self.log.important(f"Converting column {feature} to categorical (unique {unique_values})")
                    self.output_df[feature] = self.output_df[feature].astype("category")
                else:
                    self.log.warning(f"Column {feature} too many unique values {unique_values} skipping...")

    def manual_categorical_converter(self):
        """Used for Streaming: Convert object and string types to Categorical

        Note:
            This method is used for streaming/chunking. You can set the
            categorical_dtypes attribute to a dictionary of column names and
            their respective categorical types.
        """
        for column, cat_d_type in self.categorical_dtypes.items():
            self.output_df[column] = self.output_df[column].astype(cat_d_type)

    @staticmethod
    def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
        """Convert the types of the DataFrame to the correct types for the Feature Store"""
        for column in list(df.select_dtypes(include="bool").columns):
            df[column] = df[column].astype("int32")
        for column in list(df.select_dtypes(include="category").columns):
            df[column] = df[column].astype("str")

        # Select all columns that are of datetime dtype and convert them to ISO-8601 strings
        for column in [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]:
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

        # Remove any columns generated from AWS
        aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
        self.output_df = self.output_df.drop(columns=aws_cols, errors="ignore")

        # If one-hot columns are provided then one-hot encode them
        if self.one_hot_columns:
            self.output_df = self.one_hot_encode(self.output_df, self.one_hot_columns)

        # Convert columns names to lowercase, Athena will not work with uppercase column names
        if str(self.output_df.columns) != str(self.output_df.columns.str.lower()):
            for c in self.output_df.columns:
                if c != c.lower():
                    self.log.important(f"Column name {c} converted to lowercase: {c.lower()}")
            self.output_df.columns = self.output_df.columns.str.lower()

        # Check for duplicate column names in the dataframe
        if len(self.output_df.columns) != len(set(self.output_df.columns)):
            self.log.critical("Duplicate column names detected in the DataFrame")
            duplicates = self.output_df.columns[self.output_df.columns.duplicated()].tolist()
            self.log.critical(f"Duplicated columns: {duplicates}")
            raise ValueError("Duplicate column names detected in the DataFrame")

        # Make sure we have the required id and event_time columns
        self._ensure_id_column()
        self._ensure_event_time()

        # Check for a training column (Workbench uses dynamic training columns)
        if "training" in self.output_df.columns:
            self.log.important(
                """Training column detected: Since FeatureSets are read-only, Workbench creates a training view
                that can be dynamically changed. We'll use this training column to create a training view."""
            )
            self.incoming_hold_out_ids = self.output_df[~self.output_df["training"]][self.id_column].tolist()
            self.output_df = self.output_df.drop(columns=["training"])

        # We need to convert some of our column types to the correct types
        # Feature Store only supports these data types:
        # - Integral
        # - Fractional
        # - String (timestamp/datetime types need to be converted to string)
        self.output_df = self.convert_column_types(self.output_df)

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
            role_arn=self.workbench_role_arn,
            enable_online_store=True,
            table_format=self.table_format,
            tags=aws_tags,
        )

        # Ensure/wait for the feature group to be created
        self.ensure_feature_group_created(my_feature_group)
        return my_feature_group

    def pre_transform(self, **kwargs):
        """Pre-Transform: Delete any existing FeatureSet and Create the Feature Group"""
        self.delete_existing()
        self.output_feature_group = self.create_feature_group()

    def transform_impl(self):
        """Transform Implementation: Ingest the data into the Feature Group"""

        # Now we actually push the data into the Feature Group (called ingestion)
        self.log.important(f"Ingesting rows into Feature Group {self.output_uuid}...")
        ingest_manager = self.output_feature_group.ingest(self.output_df, max_workers=8, max_processes=4, wait=False)
        try:
            ingest_manager.wait()
        except IngestionError as exc:
            self.log.warning(f"Some rows had an ingesting error: {exc}")

        # Report on any rows that failed to ingest
        if ingest_manager.failed_rows:
            self.log.warning(f"Number of Failed Rows: {len(ingest_manager.failed_rows)}")

            # Log failed row details
            failed_data = self.output_df.iloc[ingest_manager.failed_rows]
            for idx, row in failed_data.iterrows():
                self.log.warning(f"Failed Row {idx}: {row.to_dict()}")

        # Keep track of the number of rows we expect to be ingested
        self.expected_rows += len(self.output_df) - len(ingest_manager.failed_rows)
        self.log.info(f"Added rows: {len(self.output_df)}")
        self.log.info(f"Failed rows: {len(ingest_manager.failed_rows)}")
        self.log.info(f"Total rows ingested: {self.expected_rows}")

        # We often need to wait a bit for AWS to fully register the new Feature Group
        self.log.important(f"Waiting for AWS to register the new Feature Group {self.output_uuid}...")
        time.sleep(30)

    def post_transform(self, **kwargs):
        """Post-Transform: Populating Offline Storage and onboard()"""
        self.log.info("Post-Transform: Populating Offline Storage and onboard()...")

        # Feature Group Ingestion takes a while, so we need to wait for it to finish
        self.output_feature_set = FeatureSetCore(self.output_uuid)
        self.log.important("Waiting for AWS Feature Group Offline storage to be ready...")
        self.log.important("This will often take 10-20 minutes...go have coffee or lunch :)")
        self.output_feature_set.set_status("initializing")
        self.wait_for_rows(self.expected_rows)

        # Call the FeatureSet onboard method to compute a bunch of EDA stuff
        self.output_feature_set.onboard()

        # Set Hold Out Ids (if we got them during creation)
        if self.incoming_hold_out_ids:
            self.output_feature_set.set_training_holdouts(self.id_column, self.incoming_hold_out_ids)

    def ensure_feature_group_created(self, feature_group):
        status = feature_group.describe().get("FeatureGroupStatus")
        while status == "Creating":
            self.log.debug("FeatureSet being Created...")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
        if status == "Created":
            self.log.info(f"FeatureSet {feature_group.name} successfully created")
        else:
            self.log.critical(f"FeatureSet {feature_group.name} creation failed with status: {status}")

    def wait_for_rows(self, expected_rows: int):
        """Wait for AWS Feature Group to fully populate the Offline Storage"""
        rows = self.output_feature_set.num_rows()

        # Wait for the rows to be populated
        self.log.info(f"Waiting for AWS Feature Group {self.output_uuid} Offline Storage...")
        max_retry = 20
        num_retry = 0
        sleep_time = 30
        while rows < expected_rows and num_retry < max_retry:
            num_retry += 1
            time.sleep(sleep_time)
            rows = self.output_feature_set.num_rows()
            self.log.info(f"Checking Offline Storage {self.output_uuid}: {rows}/{expected_rows} rows")
        if rows == expected_rows:
            self.log.important(f"Success: Reached Expected Rows ({rows} rows)...")
        else:
            msg = f"Did not reach expected rows ({rows}/{expected_rows})...(probably AWS lag)"
            self.log.warning(msg)
            self.log.monitor(msg)


if __name__ == "__main__":
    """Exercise the PandasToFeatures Class"""
    from workbench.api.data_source import DataSource

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 15)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Temp
    fs = FeatureSetCore("test_features")

    # Grab the test_data DataSource
    ds = DataSource("test_data")
    data_df = ds.sample()

    # Test setting a training column
    data_df["training"] = False
    data_df.loc[0:80, "training"] = True

    # Create my DF to Feature Set Transform (with one-hot encoding)
    df_to_features = PandasToFeatures("test_features")
    df_to_features.set_input(data_df, id_column="id", one_hot_columns=["food"])
    df_to_features.set_output_tags(["test", "small"])
    df_to_features.transform()

    # Test non-compliant output UUID
    df_to_features = PandasToFeatures("test_features-123")

    #
    # Individual Tests
    #
    """
    # Test converting columns to categorical
    df_to_features = PandasToFeatures("test_features")
    df_to_features.set_input(data_df, id_column="id")
    df_to_features.convert_columns_to_categorical(["food", "likes_dogs"])

    # Test the one-hot encoding
    df_to_features.set_input(data_df, id_column="id", one_hot_columns=["food", "likes_dogs"])
    one_hot_df = df_to_features.one_hot_encode(data_df, ["food", "likes_dogs"])
    print(one_hot_df)
    """
