"""FeatureSet: SageWorks Feature Set accessible through Athena"""

import time
from datetime import datetime, timezone

import botocore.exceptions
import pandas as pd
import awswrangler as wr
import numpy as np

from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_store import FeatureStore

# SageWorks Imports
from sageworks.core.artifacts.artifact import Artifact
from sageworks.core.artifacts.data_source_factory import DataSourceFactory
from sageworks.core.artifacts.athena_source import AthenaSource
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sageworks.core.views import View


class FeatureSetCore(Artifact):
    """FeatureSetCore: SageWorks FeatureSetCore Class

    Common Usage:
        ```
        my_features = FeatureSetCore(feature_uuid)
        my_features.summary()
        my_features.details()
        ```
    """

    def __init__(self, feature_set_uuid: str, force_refresh: bool = False):
        """FeatureSetCore Initialization

        Args:
            feature_set_uuid (str): Name of Feature Set
            force_refresh (bool): Force a refresh of the Feature Set metadata (default: False)
        """

        # Make sure the feature_set name is valid
        self.ensure_valid_name(feature_set_uuid)

        # Call superclass init
        super().__init__(feature_set_uuid)

        # Setup our AWS Broker catalog metadata
        _catalog_meta = self.aws_broker.get_metadata(ServiceCategory.FEATURE_STORE, force_refresh=force_refresh)
        self.feature_meta = _catalog_meta.get(self.uuid)

        # Sanity check and then set up our FeatureSet attributes
        if self.feature_meta is None:
            self.log.error(f"Could not find feature set {self.uuid} within current visibility scope")
            self.data_source = None
            return
        else:
            self.record_id = self.feature_meta["RecordIdentifierFeatureName"]
            self.event_time = self.feature_meta["EventTimeFeatureName"]

            # Pull Athena and S3 Storage information from metadata
            self.athena_database = self.feature_meta["sageworks_meta"].get("athena_database")
            self.athena_table = self.feature_meta["sageworks_meta"].get("athena_table")
            self.s3_storage = self.feature_meta["sageworks_meta"].get("s3_storage")

            # Create our internal DataSource (hardcoded to Athena for now)
            self.data_source = AthenaSource(self.athena_table, self.athena_database)

        # Spin up our Feature Store
        self.feature_store = FeatureStore(self.sm_session)

        # Call superclass post_init
        super().__post_init__()

        # All done
        self.log.info(f"FeatureSet Initialized: {self.uuid}...")

    def refresh_meta(self):
        """Internal: Refresh our internal AWS Feature Store metadata"""
        self.log.info("Calling refresh_meta() on the underlying DataSource")
        self.data_source.refresh_meta()

    def exists(self) -> bool:
        """Does the feature_set_name exist in the AWS Metadata?"""
        if self.feature_meta is None:
            self.log.debug(f"FeatureSet {self.uuid} not found in AWS Metadata!")
            return False
        return True

    def health_check(self) -> list[str]:
        """Perform a health check on this model

        Returns:
            list[str]: List of health issues
        """
        # Call the base class health check
        health_issues = super().health_check()

        # If we have a 'needs_onboard' in the health check then just return
        if "needs_onboard" in health_issues:
            return health_issues

        # Check our DataSource
        if not self.data_source.exists():
            self.log.critical(f"Data Source check failed for {self.uuid}")
            self.log.critical("Delete this Feature Set and recreate it to fix this issue")
            health_issues.append("data_source_missing")
        return health_issues

    def aws_meta(self) -> dict:
        """Get ALL the AWS metadata for this artifact"""
        return self.feature_meta

    def arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for this artifact"""
        return self.feature_meta["FeatureGroupArn"]

    def size(self) -> float:
        """Return the size of the internal DataSource in MegaBytes"""
        return self.data_source.size()

    @property
    def columns(self) -> list[str]:
        """Return the column names of the Feature Set"""
        return list(self.column_details().keys())

    @property
    def column_types(self) -> list[str]:
        """Return the column types of the Feature Set"""
        return list(self.column_details().values())

    def column_details(self) -> dict:
        """Return the column details of the Feature Set

        Returns:
            dict: The column details of the Feature Set

        Notes:
            We can't call just call self.data_source.column_details() because FeatureSets have different
            types, so we need to overlay that type information on top of the DataSource type information
        """
        fs_details = {item["FeatureName"]: item["FeatureType"] for item in self.feature_meta["FeatureDefinitions"]}
        ds_details = self.data_source.column_details()

        # Overlay the FeatureSet type information on top of the DataSource type information
        for col, dtype in ds_details.items():
            ds_details[col] = fs_details.get(col, dtype)
        return ds_details

    def views(self) -> list[str]:
        """Return the views for this Data Source"""
        from sageworks.core.views.view_utils import list_views

        return list_views(self.data_source)

    def view(self, view_name: str) -> "View":
        """Return a DataFrame for a specific view
        Args:
            view_name (str): The name of the view to return
        Returns:
            pd.DataFrame: A DataFrame for the specified view
        """
        from sageworks.core.views import View

        return View(self, view_name)

    def set_display_columns(self, diplay_columns: list[str]):
        """Set the display columns for this Data Source

        Args:
            diplay_columns (list[str]): The display columns for this Data Source
        """
        # Check mismatch of display columns to computation columns
        c_view = self.view("computation")
        computation_columns = c_view.columns
        mismatch_columns = [col for col in diplay_columns if col not in computation_columns]
        if mismatch_columns:
            self.log.monitor(f"Display View/Computation mismatch: {mismatch_columns}")

        self.log.important(f"Setting Display Columns...{diplay_columns}")
        from sageworks.core.views import DisplayView

        # Create a NEW display view
        DisplayView.create(self, source_table=c_view.table, column_list=diplay_columns)

    def set_computation_columns(self, computation_columns: list[str], reset_display: bool = True):
        """Set the computation columns for this FeatureSet

        Args:
            computation_columns (list[str]): The computation columns for this FeatureSet
            reset_display (bool): Also reset the display columns to match (default: True)
        """
        self.log.important(f"Setting Computation Columns...{computation_columns}")
        from sageworks.core.views import ComputationView

        # Create a NEW computation view
        ComputationView.create(self, column_list=computation_columns)
        self.recompute_stats()

        # Reset the display columns to match the computation columns
        if reset_display:
            self.set_display_columns(computation_columns)

    def num_columns(self) -> int:
        """Return the number of columns of the Feature Set"""
        return len(self.columns)

    def num_rows(self) -> int:
        """Return the number of rows of the internal DataSource"""
        return self.data_source.num_rows()

    def query(self, query: str, overwrite: bool = True) -> pd.DataFrame:
        """Query the internal DataSource

        Args:
            query (str): The query to run against the DataSource
            overwrite (bool): Overwrite the table name in the query (default: True)

        Returns:
            pd.DataFrame: The results of the query
        """
        if overwrite:
            query = query.replace(" " + self.uuid + " ", " " + self.athena_table + " ")
        return self.data_source.query(query)

    def aws_url(self):
        """The AWS URL for looking at/querying the underlying data source"""
        return self.data_source.details().get("aws_url", "unknown")

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self.feature_meta["CreationTime"]

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        # Note: We can't currently figure out how to this from AWS Metadata
        return self.feature_meta["CreationTime"]

    def get_data_source(self) -> DataSourceFactory:
        """Return the underlying DataSource object"""
        return self.data_source

    def get_feature_store(self) -> FeatureStore:
        """Return the underlying AWS FeatureStore object. This can be useful for more advanced usage
        with create_dataset() such as Joins and time ranges and a host of other options
        See: https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-create-a-dataset.html
        """
        return self.feature_store

    def create_s3_training_data(self) -> str:
        """Create some Training Data (S3 CSV) from a Feature Set using standard options. If you want
        additional options/features use the get_feature_store() method and see AWS docs for all
        the details: https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-create-a-dataset.html
        Returns:
            str: The full path/file for the CSV file created by Feature Store create_dataset()
        """

        # Set up the S3 Query results path
        date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")
        s3_output_path = self.feature_sets_s3_path + f"/{self.uuid}/datasets/all_{date_time}"

        # Make the query
        table = self.view("training").table
        query = f"SELECT * FROM {table}"
        athena_query = FeatureGroup(name=self.uuid, sagemaker_session=self.sm_session).athena_query()
        athena_query.run(query, output_location=s3_output_path)
        athena_query.wait()
        query_execution = athena_query.get_query_execution()

        # Get the full path to the S3 files with the results
        full_s3_path = s3_output_path + f"/{query_execution['QueryExecution']['QueryExecutionId']}.csv"
        return full_s3_path

    def get_training_data(self) -> pd.DataFrame:
        """Get the training data for this FeatureSet

        Returns:
            pd.DataFrame: The training data for this FeatureSet
        """
        from sageworks.core.views.view import View

        return View(self, "training").pull_dataframe()

    def snapshot_query(self, table_name: str = None) -> str:
        """An Athena query to get the latest snapshot of features

        Args:
            table_name (str): The name of the table to query (default: None)

        Returns:
            str: The Athena query to get the latest snapshot of features
        """
        # Remove FeatureGroup metadata columns that might have gotten added
        columns = self.columns
        filter_columns = ["write_time", "api_invocation_time", "is_deleted"]
        columns = ", ".join(['"' + x + '"' for x in columns if x not in filter_columns])

        query = (
            f"SELECT {columns} "
            f"    FROM (SELECT *, row_number() OVER (PARTITION BY {self.record_id} "
            f"        ORDER BY {self.event_time} desc, api_invocation_time DESC, write_time DESC) AS row_num "
            f'        FROM "{table_name}") '
            "    WHERE row_num = 1 and NOT is_deleted;"
        )
        return query

    def details(self, recompute: bool = False) -> dict[dict]:
        """Additional Details about this FeatureSet Artifact

        Args:
            recompute (bool): Recompute the details (default: False)

        Returns:
            dict(dict): A dictionary of details about this FeatureSet
        """

        # Check if we have cached version of the FeatureSet Details
        storage_key = f"feature_set:{self.uuid}:details"
        cached_details = self.data_storage.get(storage_key)
        if cached_details and not recompute:
            return cached_details

        self.log.info(f"Recomputing FeatureSet Details ({self.uuid})...")
        details = self.summary()
        details["aws_url"] = self.aws_url()

        # Store the AWS URL in the SageWorks Metadata
        self.upsert_sageworks_meta({"aws_url": details["aws_url"]})

        # Now get a summary of the underlying DataSource
        details["storage_summary"] = self.data_source.summary()

        # Number of Columns
        details["num_columns"] = self.num_columns()

        # Number of Rows
        details["num_rows"] = self.num_rows()

        # Additional Details
        details["sageworks_status"] = self.get_status()
        details["sageworks_input"] = self.get_input()
        details["sageworks_tags"] = self.tag_delimiter.join(self.get_tags())

        # Underlying Storage Details
        details["storage_type"] = "athena"  # TODO: Add RDS support
        details["storage_uuid"] = self.data_source.uuid

        # Add the column details and column stats
        details["column_details"] = self.column_details()
        details["column_stats"] = self.column_stats()

        # Cache the details
        self.data_storage.set(storage_key, details)

        # Return the details data
        return details

    def delete(self):
        """Delete the Feature Set: Feature Group, Catalog Table, and S3 Storage Objects"""

        # Delete the Feature Group and ensure that it gets deleted
        self.log.important(f"Deleting FeatureSet {self.uuid}...")
        remove_fg = FeatureGroup(name=self.uuid, sagemaker_session=self.sm_session)
        remove_fg.delete()
        self.ensure_feature_group_deleted(remove_fg)

        # Delete our underlying DataSource (Data Catalog Table and S3 Storage Objects)
        self.data_source.delete()

        # Delete any views
        self.delete_views()

        # Feature Sets can often have a lot of cruft so delete the entire bucket/prefix
        s3_delete_path = self.feature_sets_s3_path + f"/{self.uuid}/"
        self.log.info(f"Deleting All FeatureSet S3 Storage Objects {s3_delete_path}")
        wr.s3.delete_objects(s3_delete_path, boto3_session=self.boto3_session)

        # Now delete any data in the Cache
        for key in self.data_storage.list_subkeys(f"feature_set:{self.uuid}:"):
            self.log.info(f"Deleting Cache Key: {key}")
            self.data_storage.delete(key)

        # Force a refresh of the AWS Metadata (to make sure references to deleted artifacts are gone)
        self.aws_broker.get_metadata(ServiceCategory.FEATURE_STORE, force_refresh=True)

    def ensure_feature_group_deleted(self, feature_group):
        status = "Deleting"
        while status == "Deleting":
            self.log.debug("FeatureSet being Deleted...")
            try:
                status = feature_group.describe().get("FeatureGroupStatus")
            except botocore.exceptions.ClientError as error:
                # For ResourceNotFound/ValidationException, this is fine, otherwise raise all other exceptions
                if error.response["Error"]["Code"] in ["ResourceNotFound", "ValidationException"]:
                    break
                else:
                    raise error
            time.sleep(1)
        self.log.info(f"FeatureSet {feature_group.name} successfully deleted")

    def set_training_holdouts(self, id_column: str, holdout_ids: list[str]):
        """Set the hold out ids for the training view for this FeatureSet

        Args:
            id_column (str): The name of the id column.
            holdout_ids (list[str]): The list of holdout ids.
        """
        from sageworks.core.views import TrainingView

        # Create a NEW training view
        TrainingView.create(self, id_column=id_column, holdout_ids=holdout_ids)

    def delete_views(self):
        """Delete any views associated with this FeatureSet"""
        from sageworks.core.views.view_utils import delete_views_and_supplemental_data

        delete_views_and_supplemental_data(self.data_source)

    def descriptive_stats(self, recompute: bool = False) -> dict:
        """Get the descriptive stats for the numeric columns of the underlying DataSource
        Args:
            recompute (bool): Recompute the descriptive stats (default=False)
        Returns:
            dict: A dictionary of descriptive stats for the numeric columns
        """
        return self.data_source.descriptive_stats(recompute)

    def sample(self, recompute: bool = False) -> pd.DataFrame:
        """Get a sample of the data from the underlying DataSource
        Args:
            recompute (bool): Recompute the sample (default=False)
        Returns:
            pd.DataFrame: A sample of the data from the underlying DataSource
        """
        return self.data_source.sample(recompute)

    def outliers(self, scale: float = 1.5, recompute: bool = False) -> pd.DataFrame:
        """Compute outliers for all the numeric columns in a DataSource
        Args:
            scale (float): The scale to use for the IQR (default: 1.5)
            recompute (bool): Recompute the outliers (default: False)
        Returns:
            pd.DataFrame: A DataFrame of outliers from this DataSource
        Notes:
            Uses the IQR * 1.5 (~= 2.5 Sigma) method to compute outliers
            The scale parameter can be adjusted to change the IQR multiplier
        """
        return self.data_source.outliers(scale=scale, recompute=recompute)

    def smart_sample(self, recompute: bool = False) -> pd.DataFrame:
        """Get a SMART sample dataframe from this FeatureSet

        Args:
            recompute (bool): Recompute the smart sample (default=False)

        Returns:
            pd.DataFrame: A combined DataFrame of sample data + outliers
        """
        return self.data_source.smart_sample(recompute=recompute)

    def anomalies(self) -> pd.DataFrame:
        """Get a set of anomalous data from the underlying DataSource
        Returns:
            pd.DataFrame: A dataframe of anomalies from the underlying DataSource
        """

        # FIXME: Mock this for now
        anom_df = self.sample().copy()
        anom_df["anomaly_score"] = np.random.rand(anom_df.shape[0])
        anom_df["cluster"] = np.random.randint(0, 10, anom_df.shape[0])
        anom_df["x"] = np.random.rand(anom_df.shape[0])
        anom_df["y"] = np.random.rand(anom_df.shape[0])
        return anom_df

    def value_counts(self, recompute: bool = False) -> dict:
        """Get the value counts for the string columns of the underlying DataSource
        Args:
            recompute (bool): Recompute the value counts (default=False)
        Returns:
            dict: A dictionary of value counts for the string columns
        """
        return self.data_source.value_counts(recompute)

    def correlations(self, recompute: bool = False) -> dict:
        """Get the correlations for the numeric columns of the underlying DataSource
        Args:
            recompute (bool): Recompute the value counts (default=False)
        Returns:
            dict: A dictionary of correlations for the numeric columns
        """
        return self.data_source.correlations(recompute)

    def column_stats(self, recompute: bool = False) -> dict[dict]:
        """Compute Column Stats for all the columns in the FeatureSets underlying DataSource
        Args:
            recompute (bool): Recompute the column stats (default: False)
        Returns:
            dict(dict): A dictionary of stats for each column this format
            NB: String columns will NOT have num_zeros and descriptive_stats
             {'col1': {'dtype': 'string', 'unique': 4321, 'nulls': 12},
              'col2': {'dtype': 'int', 'unique': 4321, 'nulls': 12, 'num_zeros': 100, 'descriptive_stats': {...}},
              ...}
        """

        # Grab the column stats from our DataSource
        ds_column_stats = self.data_source.column_stats(recompute)

        # Map the types from our DataSource to the FeatureSet types
        fs_type_mapper = self.column_details()
        for col, details in ds_column_stats.items():
            details["fs_dtype"] = fs_type_mapper.get(col, "unknown")

        return ds_column_stats

    def ready(self) -> bool:
        """Is the FeatureSet ready? Is initial setup complete and expected metadata populated?
        Note: Since FeatureSet is a composite of DataSource and FeatureGroup, we need to
           check both to see if the FeatureSet is ready."""

        # Check the expected metadata for the FeatureSet
        expected_meta = self.expected_meta()
        existing_meta = self.sageworks_meta()
        feature_set_ready = set(existing_meta.keys()).issuperset(expected_meta)
        if not feature_set_ready:
            self.log.info(f"FeatureSet {self.uuid} is not ready!")
            return False

        # Okay now call/return the DataSource ready() method
        return self.data_source.ready()

    def onboard(self) -> bool:
        """This is a BLOCKING method that will onboard the FeatureSet (make it ready)"""

        # Set our status to onboarding
        self.log.important(f"Onboarding {self.uuid}...")
        self.set_status("onboarding")
        self.remove_health_tag("needs_onboard")

        # Call our underlying DataSource onboard method
        self.data_source.refresh_meta()
        if not self.data_source.exists():
            self.log.critical(f"Data Source check failed for {self.uuid}")
            self.log.critical("Delete this Feature Set and recreate it to fix this issue")
            return False
        if not self.data_source.ready():
            self.data_source.onboard()

        # Run a health check and refresh the meta
        time.sleep(2)  # Give the AWS Metadata a chance to update
        self.health_check()
        self.refresh_meta()
        self.details(recompute=True)
        self.set_status("ready")
        return True

    def recompute_stats(self) -> bool:
        """This is a BLOCKING method that will recompute the stats for the FeatureSet"""

        # Call our underlying DataSource recompute stats method
        self.log.important(f"Recomputing Stats {self.uuid}...")
        self.data_source.recompute_stats()
        self.details(recompute=True)
        return True


if __name__ == "__main__":
    """Exercise for FeatureSet Class"""
    from sageworks.core.artifacts.feature_set_core import FeatureSetCore as LocalFeatureSetCore  # noqa: F811
    from pprint import pprint

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Test new recompute_stats method (we're grabbing a datasource from the FeatureSet)
    fs = LocalFeatureSetCore("abalone_features")
    fs.recompute_stats()

    # Grab a FeatureSet object and pull some information from it
    my_features = LocalFeatureSetCore("test_features")

    # Call the various methods
    # What's my AWS ARN and URL
    print(f"AWS ARN: {my_features.arn()}")
    print(f"AWS URL: {my_features.aws_url()}")

    # Let's do a check/validation of the feature set
    print(f"Feature Set Check: {my_features.exists()}")

    # How many rows and columns?
    num_rows = my_features.num_rows()
    num_columns = my_features.num_columns()
    print(f"Rows: {num_rows} Columns: {num_columns}")

    # What are the column names?
    print(my_features.columns)

    # Get the metadata and tags associated with this feature set
    print(f"SageWorks Meta: {my_features.sageworks_meta()}")
    print(f"SageWorks Tags: {my_features.get_tags()}")

    # Get a summary for this Feature Set
    print("\nSummary:")
    pprint(my_features.summary())

    # Get the details for this Feature Set
    print("\nDetails:")
    pprint(my_features.details())

    # Now do deep dive on storage
    storage = my_features.get_data_source()
    print("\nStorage Details:")
    pprint(storage.details())

    # Get a sample of the data
    df = my_features.sample()
    print(f"Sample Data: {df.shape}")
    print(df)

    # Get descriptive stats for all the columns
    stat_info = my_features.descriptive_stats()
    print("Descriptive Stats")
    pprint(stat_info)

    # Get outliers for all the columns
    outlier_df = my_features.outliers()
    print(outlier_df)

    # Set the holdout ids for the training view
    print("Setting hold out ids...")
    table = my_features.view("training").table
    df = my_features.query(f"SELECT id, name FROM {table}")
    my_holdout_ids = [id for id in df["id"] if id < 20]
    my_features.set_training_holdouts("id", my_holdout_ids)

    # Test the hold out set functionality with strings
    print("Setting hold out ids (strings)...")
    my_holdout_ids = [name for name in df["name"] if int(name.split(" ")[1]) > 80]
    my_features.set_training_holdouts("name", my_holdout_ids)

    # Get the training data
    print("Getting the training data...")
    training_data = my_features.get_training_data()

    # Now delete the AWS artifacts associated with this Feature Set
    # print('Deleting SageWorks Feature Set...')
    # my_features.delete()
    print("Done")
