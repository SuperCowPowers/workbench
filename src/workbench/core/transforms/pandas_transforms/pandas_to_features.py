"""PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet"""

import pandas as pd
import time
from sagemaker.core.resources import FeatureGroup
from sagemaker.core.shapes.shapes import OnlineStoreConfig, OfflineStoreConfig, S3StorageConfig
from sagemaker.mlops.feature_store import (
    TableFormatEnum,
    IngestionError,
    load_feature_definitions_from_dataframe,
    ingest_dataframe,
)

# Local imports
from workbench.utils.datetime_utils import datetime_to_iso8601
from workbench.utils import feature_prep
from workbench.core.transforms.transform import Transform, TransformInput, TransformOutput
from workbench.core.artifact import Artifact
from workbench.core.artifacts.feature_set_core import FeatureSetCore


class PandasToFeatures(Transform):
    """PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet

    Common Usage:
        ```python
        to_features = PandasToFeatures(output_name)
        to_features.set_output_tags(["my", "awesome", "data"])
        to_features.set_input(df, id_column="my_id", input_name="my_data_source")
        to_features.transform()
        ```
    """

    def __init__(self, output_name: str):
        """PandasToFeatures Initialization

        Args:
            output_name (str): The Name of the FeatureSet to create
        """

        # Make sure the output_name is a valid name
        Artifact.is_name_valid(output_name)

        # Call superclass init
        super().__init__("DataFrame", output_name)

        # Set up all my instance attributes
        self.input_type = TransformInput.PANDAS_DF
        self.output_type = TransformOutput.FEATURE_SET
        self.id_column = None
        self.event_time_column = None
        self.one_hot_columns = []
        self.categorical_dtypes = {}  # Used for streaming/chunking
        self.output_df = None
        self.table_format = TableFormatEnum.ICEBERG

        # These will be set in the transform method
        self.output_feature_group = None
        self.output_feature_set = None
        self.expected_rows = 0

    def set_input(
        self,
        input_df: pd.DataFrame,
        id_column=None,
        event_time_column=None,
        one_hot_columns=None,
        input_name: str = "DataFrame",
    ):
        """Set the Input DataFrame for this Transform

        Args:
            input_df (pd.DataFrame): The input DataFrame.
            id_column (str, optional): The ID column (use "auto"/None for auto-generated IDs).
            event_time_column (str, optional): The name of the event time column (default: None).
            one_hot_columns (list, optional): The list of columns to one-hot encode (default: None).
            input_name (str, optional): The artifact this DataFrame came from, recorded as the
                                        FeatureSet's provenance (default: "DataFrame").
        """
        self.id_column = id_column
        self.event_time_column = event_time_column
        self.output_df = input_df.copy()
        self.one_hot_columns = one_hot_columns or []
        self.input_name = input_name
        self.output_meta["workbench_input"] = input_name

        # Warn about known AWS Iceberg bug with event_time_column
        if event_time_column is not None:
            self.log.warning(
                f"event_time_column='{event_time_column}' specified. Note: AWS has a known bug with "
                "Iceberg FeatureGroups where varying event times across multiple days can cause "
                "duplicate rows in the offline store. Setting event_time_column=None."
            )
            self.event_time_column = None

        # Now Prepare the DataFrame for its journey into an AWS FeatureGroup
        self.prep_dataframe()

    def delete_existing(self):
        # Delete the existing FeatureSet if it exists
        self.log.info(f"Deleting the {self.output_name} FeatureSet...")
        FeatureSetCore.managed_delete(self.output_name)
        time.sleep(1)

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

    def prep_dataframe(self):
        """Prep the DataFrame for Feature Store Creation"""
        self.output_df, self.id_column = feature_prep.prep_dataframe(
            self.output_df,
            id_column=self.id_column,
            event_time_column=self.event_time_column,
            one_hot_columns=self.one_hot_columns,
        )

        # AWS Feature Store requires an event_time field on every record
        self._ensure_event_time()

    def create_feature_group(self):
        """Create a Feature Group, load our Feature Definitions, and wait for it to be ready"""

        # Load Feature Definitions from the DataFrame
        feature_definitions = load_feature_definitions_from_dataframe(self.output_df)

        # Create the Output S3 Storage Path for this Feature Set
        s3_storage_path = f"{self.feature_sets_s3_path}/{self.output_name}"

        # Get the metadata/tags to push into AWS
        aws_tags = self.get_aws_tags()

        # Create the Feature Group using V3 resource class
        my_feature_group = FeatureGroup.create(
            feature_group_name=self.output_name,
            record_identifier_feature_name=self.id_column,
            event_time_feature_name=self.event_time_column,
            feature_definitions=feature_definitions,
            online_store_config=OnlineStoreConfig(enable_online_store=True),
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=s3_storage_path),
                table_format=self.table_format,
            ),
            role_arn=self.workbench_role_arn,
            tags=aws_tags,
            session=self.boto3_session,
        )

        # Ensure/wait for the feature group to be created
        self.ensure_feature_group_created(my_feature_group)
        return my_feature_group

    def pre_transform(self, **kwargs):
        """Pre-Transform: Delete any existing FeatureSet and Create the Feature Group"""
        self.delete_existing()
        self.output_feature_group = self.create_feature_group()

    @staticmethod
    def _ingest_settings():
        """Return (max_workers, max_processes) based on multiprocessing mode.

        SageMaker V3's _run_multi_process defines a local function (init_worker) that can't be
        pickled by spawn/forkserver workers, so any value > 1 crashes there. Fork workers inherit
        the initializer in child memory, so multiprocessing can stay enabled on Linux/AWS Batch.
        See: https://github.com/aws/sagemaker-python-sdk/issues/5312
        """
        import multiprocessing

        if multiprocessing.get_start_method() != "fork":
            return 1, 1
        return 8, 4

    def transform_impl(self):
        """Transform Implementation: Ingest the data into the Feature Group"""

        max_workers, max_processes = self._ingest_settings()
        self.log.important(f"Ingesting rows into Feature Group {self.output_name}...")
        self.log.info(f"Ingest settings: max_workers={max_workers}, max_processes={max_processes}")
        failed_rows = []
        try:
            ingest_dataframe(
                feature_group_name=self.output_name,
                data_frame=self.output_df,
                max_workers=max_workers,
                max_processes=max_processes,
                wait=True,
            )
        except IngestionError as exc:
            failed_rows = exc.failed_rows
            self.log.warning(f"Some rows had an ingesting error: {exc}")

        # Keep track of the number of rows we expect to be ingested
        self.expected_rows += len(self.output_df) - len(failed_rows)
        self.log.info(f"Added rows: {len(self.output_df)}")
        self.log.info(f"Failed rows: {len(failed_rows)}")
        self.log.info(f"Total rows ingested: {self.expected_rows}")

        # We often need to wait a bit for AWS to fully register the new Feature Group
        self.log.important(f"Waiting for AWS to register the new Feature Group {self.output_name}...")
        time.sleep(30)

    def post_transform(self, **kwargs):
        """Post-Transform: Populating Offline Storage and onboard()"""
        self.log.info("Post-Transform: Populating Offline Storage and onboard()...")

        # Feature Group Ingestion takes a while, so we need to wait for it to finish
        self.output_feature_set = FeatureSetCore(self.output_name)
        self.log.important("Waiting for AWS Feature Group Offline storage to be ready...")
        self.log.important("This will often take 10-20 minutes...go have coffee or lunch :)")
        self.output_feature_set.set_status("initializing")
        self.wait_for_rows(self.expected_rows)

        # Call the FeatureSet onboard method to compute a bunch of EDA stuff
        self.output_feature_set.onboard()

    def ensure_feature_group_created(self, feature_group):
        feature_group.refresh()
        status = feature_group.feature_group_status
        while status == "Creating":
            self.log.debug("FeatureSet being Created…")
            time.sleep(5)
            feature_group.refresh()
            status = feature_group.feature_group_status

        if status == "Created":
            self.log.info(f"FeatureSet {feature_group.get_name()} successfully created")
        else:
            failure_reason = getattr(feature_group, "failure_reason", "No failure reason provided")
            self.log.critical(f"FeatureSet {feature_group.get_name()} creation failed with status: {status}")
            self.log.critical(f"Failure reason: {failure_reason}")

    def wait_for_rows(self, expected_rows: int):
        """Wait for AWS Feature Group to fully populate the Offline Storage"""
        rows = self.output_feature_set.num_rows()

        # Wait for the rows to be populated
        self.log.info(f"Waiting for AWS Feature Group {self.output_name} Offline Storage...")
        max_retry = 20
        num_retry = 0
        sleep_time = 30
        while rows < expected_rows and num_retry < max_retry:
            num_retry += 1
            time.sleep(sleep_time)
            rows = self.output_feature_set.num_rows()
            self.log.info(f"Checking Offline Storage {self.output_name}: {rows}/{expected_rows} rows")
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
    FeatureSetCore("test_features")

    # Grab the test_data DataSource
    ds = DataSource("test_data")
    data_df = ds.sample()

    # Create my DF to Feature Set Transform (with one-hot encoding)
    df_to_features = PandasToFeatures("test_features")
    df_to_features.set_input(data_df, id_column="id", event_time_column="date", one_hot_columns=["food"])
    df_to_features.set_output_tags(["test", "small"])
    df_to_features.transform()

    # Test non-compliant output Name
    PandasToFeatures("test_features-123")

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
