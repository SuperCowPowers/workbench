"""PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet"""
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import time
import botocore
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.inputs import TableFormatEnum

# Local imports
from sageworks.utils.iso_8601 import datetime_to_iso8601
from sageworks.transforms.transform import Transform
from sageworks.artifacts.feature_sets.feature_set import FeatureSet


class PandasToFeatures(Transform):
    """PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet

    Common Usage:
        to_features = PandasToFeatures(output_uuid)
        to_features.set_output_tags(["abalone", "public", "whatever"])
        to_features.set_input(df, id_column="id"/None, event_time_column="date"/None)
        to_features.transform()
    """

    def __init__(self, output_uuid: str, auto_categorical=True):
        """PandasToFeatures Initialization
        Args:
            output_uuid (str): The UUID of the FeatureSet to create
            auto_categorical (bool): Should we automatically create categorical columns?
        """
        # Call superclass init
        super().__init__("DataFrame", output_uuid)

        # Set up all my instance attributes
        self.id_column = None
        self.event_time_column = None
        self.auto_categorical = auto_categorical
        self.categorical_dtypes = {}
        self.output_df = None
        self.table_format = TableFormatEnum.ICEBERG

        # Delete the existing FeatureSet if it exists
        self.delete_existing()

        # These will be set in the transform method
        self.output_feature_group = None
        self.output_feature_set = None
        self.expected_rows = 0

    def set_input(self, input_df: pd.DataFrame, id_column=None, event_time_column=None):
        """Set the Input DataFrame for this Transform"""
        self.id_column = id_column
        self.event_time_column = event_time_column
        self.output_df = input_df.copy()

        # Now Prepare the DataFrame for its journey into an AWS FeatureGroup
        self.prep_dataframe()

    def delete_existing(self):
        # Delete the existing FeatureSet if it exists
        try:
            delete_fs = FeatureSet(self.output_uuid)
            if delete_fs.check():
                delete_fs.delete()
                self.log.info(f"Deleted the {self.output_uuid} FeatureSet...")
        except botocore.exceptions.ClientError as exc:
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
            # current_datetime = datetime.now(timezone.utc)
            self.log.info("Generating an event_time column before FeatureSet Creation...")
            self.event_time_column = "event_time"
            self.output_df[self.event_time_column] = pd.Timestamp("now", tz="UTC")

        # The event_time_column is defined so lets make sure it the right type for Feature Store
        if pd.api.types.is_datetime64_any_dtype(self.output_df[self.event_time_column]):
            self.log.info(f"Converting {self.event_time_column} to ISOFormat Date String before FeatureSet Creation...")

            # Convert the datetime DType to ISO-8601 string
            self.output_df[self.event_time_column] = self.output_df[self.event_time_column].map(datetime_to_iso8601)
            self.output_df[self.event_time_column] = self.output_df[self.event_time_column].astype(pd.StringDtype())

    def _convert_objs_to_string(self):
        """Internal: AWS Feature Store doesn't know how to store object dtypes, so convert to String"""
        for col in self.output_df:
            if pd.api.types.is_object_dtype(self.output_df[col].dtype):
                self.output_df[col] = self.output_df[col].astype(pd.StringDtype())

    # Helper Methods
    def auto_categorical_converter(self):
        """Convert object and string types to Categorical"""
        categorical_columns = []
        for feature, dtype in self.output_df.dtypes.items():
            print(feature, dtype)
            if dtype in ["object", "string"] and feature not in [self.event_time_column, self.id_column]:
                unique_values = self.output_df[feature].nunique()
                print(f"Unique Values = {unique_values}")
                if unique_values < 5:
                    print(f"Converting object column {feature} to categorical")
                    self.output_df[feature] = self.output_df[feature].astype("category")
                    categorical_columns.append(feature)

        # Now convert Categorical Types to One Hot Encoding
        self.output_df = pd.get_dummies(self.output_df, columns=categorical_columns)

    def manual_categorical_converter(self):
        """Convert object and string types to Categorical"""
        for column, cat_d_type in self.categorical_dtypes.items():
            self.output_df[column] = self.output_df[column].astype(cat_d_type)

        # Now convert Categorical Types to One Hot Encoding
        categorical_columns = list(self.categorical_dtypes.keys())
        self.output_df = pd.get_dummies(self.output_df, columns=categorical_columns)

    @staticmethod
    def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
        """Convert the types of the DataFrame to the correct types for the Feature Store"""
        datetime_type = ["datetime", "datetime64", "datetime64[ns]", "datetimetz"]
        for column in df.select_dtypes(include=datetime_type).columns:
            df[column] = df[column].astype("string")
        for column in list(df.select_dtypes(include="object").columns):
            df[column] = df[column].astype("string")
        for column in list(df.select_dtypes(include=[pd.Int64Dtype]).columns):
            df[column] = df[column].astype("int64")
        for column in list(df.select_dtypes(include="bool").columns):
            df[column] = df[column].astype("int64")
        for column in list(df.select_dtypes(include=[pd.Float64Dtype]).columns):
            df[column] = df[column].astype("float64")
        return df

    def prep_dataframe(self):
        """Prep the DataFrame for Feature Store Creation"""
        self.log.info("Prep the output_df (cat_convert, convert types, lowercase columns, add training column)...")

        # Make sure we have the required id and event_time columns
        self._ensure_id_column()
        self._ensure_event_time()

        # Convert object and string types to Categorical
        if self.auto_categorical:
            self.auto_categorical_converter()
        else:
            self.manual_categorical_converter()

        # We need to convert some of our column types to the correct types
        # Feature Store only supports these data types:
        # - Integral
        # - Fractional
        # - String (timestamp/datetime types need to be converted to string)
        self.output_df = self.convert_column_types(self.output_df)

        # FeatureSet Internal Storage (Athena) will convert columns names to lowercase, so we need
        # to make sure that the column names are lowercase to match and avoid downstream issues
        self.output_df.columns = self.output_df.columns.str.lower()

        # Mark 80% of the data as training and 20% as validation/test
        self.output_df["training"] = np.random.binomial(size=len(self.output_df), n=1, p=0.8)

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

        # Now we actually push the data into the Feature Group
        ingest_manager = self.output_feature_group.ingest(self.output_df, max_processes=4, wait=False)
        ingest_manager.wait()

        # Report on any rows that failed to ingest
        if ingest_manager.failed_rows:
            self.log.info(f"Failed Rows: {ingest_manager.failed_rows}")

        # Keep track of the number of rows we expect to be ingested
        self.expected_rows += len(self.output_df) - len(ingest_manager.failed_rows)
        self.log.info(f"Added rows: {len(self.output_df)}")
        self.log.info(f"Failed rows: {len(ingest_manager.failed_rows)}")
        self.log.info(f"Total rows to be ingested: {self.expected_rows}")

    def post_transform(self, **kwargs):
        """Post-Transform: Compute the Details, Quartiles, and SampleDF for the FeatureSet"""
        self.log.info("Post-Transform: Computing Details, Quartiles, and SampleDF for the FeatureSet...")

        # Feature Group Ingestion takes a while, so we need to wait for it to finish
        self.output_feature_set = FeatureSet(self.output_uuid, force_refresh=True)
        self.output_feature_set.set_status("initializing")

        # Wait for offline storage of the Feature Group to be ready
        self.log.info("Waiting for Feature Group Offline storage to be ready...")
        self.log.info("Note: This will often take 10-20 minutes...go have coffee or lunch :)")
        self.wait_for_rows(self.expected_rows)

        # Now compute the Details, Quartiles, and SampleDF for the FeatureSet
        self.output_feature_set.details()
        self.output_feature_set.data_source.details()
        self.output_feature_set.value_counts()
        self.output_feature_set.quartiles()
        self.output_feature_set.sample_df()
        self.output_feature_set.set_status("ready")

    def ensure_feature_group_created(self, feature_group):
        status = feature_group.describe().get("FeatureGroupStatus")
        while status == "Creating":
            self.log.info("FeatureSet being Created...")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
        self.log.info(f"FeatureSet {feature_group.name} successfully created")

    def wait_for_rows(self, expected_rows: int):
        """Wait for AWS Feature Group to fully populate the Offline Storage"""
        rows = self.output_feature_set.num_rows()
        while rows < expected_rows:
            self.log.info(f"Waiting for AWS Feature Group {self.output_uuid} Offline Storage ({rows} rows)...")
            sleep_time = 5 if rows else 30
            time.sleep(sleep_time)
            rows = self.output_feature_set.num_rows()
        self.log.info(f"Success: Reached Expected Rows ({rows} rows)...")


if __name__ == "__main__":
    """Exercise the PandasToFeatures Class"""

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 15)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Create some fake data
    my_datetime = datetime.now(timezone.utc)
    fake_data = [
        {"id": 1, "name": "sue", "age": 41, "score": 7.8, "date": my_datetime, "cat": "a"},
        {"id": 2, "name": "bob", "age": 34, "score": 6.4, "date": my_datetime, "cat": "b"},
        {"id": 3, "name": "ted", "age": 69, "score": 8.2, "date": my_datetime, "cat": "c"},
        {"id": 4, "name": "bill", "age": 24, "score": 5.3, "date": my_datetime, "cat": "a"},
        {"id": 5, "name": "sally", "age": 52, "score": 9.5, "date": my_datetime, "cat": "b"},
    ]
    fake_df = pd.DataFrame(fake_data)

    # Create my DF to Feature Set Transform
    df_to_features = PandasToFeatures("test_feature_set")
    df_to_features.set_input(fake_df, id_column="id", event_time_column="date")
    df_to_features.set_output_tags(["test", "small"])

    # Store this dataframe as a SageWorks Feature Set
    df_to_features.transform()
