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
from sageworks.artifacts.artifact import Artifact
from sageworks.artifacts.data_sources.data_source import DataSource
from sageworks.artifacts.data_sources.athena_source import AthenaSource
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory
from sageworks.utils.iso_8601 import convert_all_to_iso8601


class FeatureSet(Artifact):
    """FeatureSet: SageWorks Feature Set accessible through Athena

    Common Usage:
        my_features = FeatureSet(feature_uuid)
        my_features.summary()
        my_features.details()
    """

    def __init__(self, feature_set_uuid: str, force_refresh: bool = False):
        """FeatureSet Initialization
        Args:
            feature_set_uuid (str): Name of Feature Set in SageWorks Metadata
            force_refresh (bool): Force a refresh of the Feature Set metadata (default: False)
        """
        # Call superclass init
        super().__init__(feature_set_uuid)

        # Setup our AWS Broker catalog metadata
        _catalog_meta = self.aws_broker.get_metadata(ServiceCategory.FEATURE_STORE, force_refresh=force_refresh)
        self.feature_meta = _catalog_meta.get(self.uuid)

        # Sanity check and then set up our FeatureSet attributes
        if self.feature_meta is None:
            self.log.info(f"Could not find feature set {self.uuid} within current visibility scope")
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

        # All done
        self.log.info(f"FeatureSet Initialized: {self.uuid}")

    def refresh_meta(self):
        """Internal: Refresh our internal AWS Feature Store metadata"""
        self.log.warning("Calling refresh_meta() on the underlying DataSource")
        self.data_source.refresh_meta()

    def exists(self) -> bool:
        """Does the feature_set_name exist in the AWS Metadata?"""
        if self.feature_meta is None:
            self.log.info(f"FeatureSet.exists() {self.uuid} not found in AWS Metadata!")
            return False
        else:  # Also check our Data Source
            if not self.data_source.exists():
                self.log.critical(f"Data Source check failed for {self.uuid}")
                self.log.critical("Delete this Feature Set and recreate it to fix this issue")
                return False
        # AOK
        return True

    def aws_meta(self) -> dict:
        """Get ALL the AWS metadata for this artifact"""
        return self.feature_meta

    def arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for this artifact"""
        return self.feature_meta["FeatureGroupArn"]

    def size(self) -> float:
        """Return the size of the internal DataSource in MegaBytes"""
        return self.data_source.size()

    def column_names(self) -> list[str]:
        """Return the column names of the Feature Set"""
        return list(self.column_details().keys())

    def column_types(self) -> list[str]:
        """Return the column types of the Feature Set"""
        return list(self.column_details().values())

    def column_details(self) -> dict:
        """Return the column details of the Feature Set"""
        details = {item["FeatureName"]: item["FeatureType"] for item in self.feature_meta["FeatureDefinitions"]}
        internal = {
            "write_time": "Timestamp",
            "api_invocation_time": "Timestamp",
            "is_deleted": "Boolean",
        }
        details.update(internal)
        return details

    def num_columns(self) -> int:
        """Return the number of columns of the Feature Set"""
        return len(self.column_names())

    def num_rows(self) -> int:
        """Return the number of rows of the internal DataSource"""
        return self.data_source.num_rows()

    def query(self, query: str) -> pd.DataFrame:
        """Query the internal DataSource"""
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

    def get_data_source(self) -> DataSource:
        """Return the underlying DataSource object"""
        return self.data_source

    def get_feature_store(self) -> FeatureStore:
        """Return the underlying AWS FeatureStore object. This can be useful for more advanced usage
        with create_dataset() such as Joins and time ranges and a host of other options
        See: https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-create-a-dataset.html
        """
        return self.feature_store

    def create_s3_training_data(self, training_split=80, test_split=20, val_split=0) -> str:
        """Create some Training Data (S3 CSV) from a Feature Set using standard options. If you want
        additional options/features use the get_feature_store() method and see AWS docs for all
        the details: https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-create-a-dataset.html
        Args:
            training_split (int): Percentage of data that goes into the TRAINING set
            test_split (int): Percentage of data that goes into the TEST set
            val_split (int): Percentage of data that goes into the VALIDATION set (default=0)
        Returns:
            str: The full path/file for the CSV file created by Feature Store create_dataset()
        """

        # Set up the S3 Query results path
        date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")
        s3_output_path = self.feature_sets_s3_path + f"/{self.uuid}/datasets/all_{date_time}"

        # Get the snapshot query
        query = self.snapshot_query()

        # Make the query
        athena_query = FeatureGroup(name=self.uuid, sagemaker_session=self.sm_session).athena_query()
        athena_query.run(query, output_location=s3_output_path)
        self.log.info("Waiting for Athena Query...")
        athena_query.wait()
        query_execution = athena_query.get_query_execution()

        # Get the full path to the S3 files with the results
        full_s3_path = s3_output_path + f"/{query_execution['QueryExecution']['QueryExecutionId']}.csv"
        return full_s3_path

    def snapshot_query(self):
        """An Athena query to get the latest snapshot of features"""
        # Remove FeatureGroup metadata columns that might have gotten added
        columns = self.column_names()
        filter_columns = ["write_time", "api_invocation_time", "is_deleted"]
        columns = ", ".join(['"' + x + '"' for x in columns if x not in filter_columns])

        query = (
            f"SELECT {columns} "
            f"    FROM (SELECT *, row_number() OVER (PARTITION BY {self.record_id} "
            f"        ORDER BY {self.event_time} desc, api_invocation_time DESC, write_time DESC) AS row_num "
            f'        FROM "{self.athena_table}") '
            "    WHERE row_num = 1 and  NOT is_deleted;"
        )
        return query

    def details(self, recompute: bool = False) -> dict[dict]:
        """Additional Details about this FeatureSet Artifact
        Args:
            recompute(bool): Recompute the details (default: False)
        Returns:
            dict(dict): A dictionary of details about this FeatureSet
        """
        # First check if we have already computed the details
        if self.sageworks_meta().get("details_computed") and not recompute:
            # Get the SageWorks Metadata
            details = self.sageworks_meta()

            # Hack for AWS URL
            details["aws_url"] = details["aws_url"].replace("__question__", "?").replace("__pound__", "#")

            # Now we need to get the details from the underlying DataSource
            details["storage_details"] = self.data_source.details()

            return details

        # Create a dictionary of details
        details = dict()
        details["details_computed"] = True

        # Number of Columns
        details["num_columns"] = self.num_columns()

        # Number of Rows
        details["num_rows"] = self.num_rows()

        # Additional Details
        details["sageworks_status"] = self.get_status()
        details["sageworks_input"] = self.get_input()
        details["sageworks_tags"] = ":".join(self.sageworks_tags())

        # Underlying Storage Details
        details["storage_type"] = "athena"  # TODO: Add RDS support
        details["storage_uuid"] = self.data_source.uuid

        # These details will be stored in AWS (as tags). AWS tags have
        # a bunch of constraints so we need to do some replacements
        details["aws_url"] = self.aws_url().replace("?", "__question__").replace("#", "__pound__")

        # Convert any datetime fields to ISO-8601 strings
        details = convert_all_to_iso8601(details)

        # Push the details data into our DataSource Metadata
        self.upsert_sageworks_meta(details)

        # Return the details data
        return details

    def delete(self):
        """Delete the Feature Set: Feature Group, Catalog Table, and S3 Storage Objects"""

        # Delete the Feature Group and ensure that it gets deleted
        remove_fg = FeatureGroup(name=self.uuid, sagemaker_session=self.sm_session)
        remove_fg.delete()
        self.ensure_feature_group_deleted(remove_fg)

        # Delete our underlying DataSource (Data Catalog Table and S3 Storage Objects)
        self.data_source.delete()

        # Feature Sets can often have a lot of cruft so delete the entire bucket/prefix
        s3_delete_path = self.feature_sets_s3_path + f"/{self.uuid}"
        self.log.info(f"Deleting S3 Storage Objects {s3_delete_path}")
        wr.s3.delete_objects(s3_delete_path, boto3_session=self.boto_session)

    def ensure_feature_group_deleted(self, feature_group):
        status = "Deleting"
        while status == "Deleting":
            self.log.info("FeatureSet being Deleted...")
            try:
                status = feature_group.describe().get("FeatureGroupStatus")
            except botocore.exceptions.ClientError as error:
                # If the exception is a ResourceNotFound, this is fine, otherwise raise all other exceptions
                if error.response["Error"]["Code"] == "ResourceNotFound":
                    break
                else:
                    raise error
            time.sleep(1)
        self.log.info(f"FeatureSet {feature_group.name} successfully deleted")

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
            scale(float): The scale to use for the IQR (default: 1.5)
            recompute(bool): Recompute the outliers (default: False)
        Returns:
            pd.DataFrame: A DataFrame of outliers from this DataSource
        Notes:
            Uses the IQR * 1.5 (~= 2.5 Sigma) method to compute outliers
            The scale parameter can be adjusted to change the IQR multiplier
        """
        return self.data_source.outliers(scale=scale, recompute=recompute)

    def smart_sample(self) -> pd.DataFrame:
        """Get a SMART sample dataframe from this FeatureSet
        Returns:
            pd.DataFrame: A combined DataFrame of sample data + outliers
        """
        return self.data_source.smart_sample()

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
            recompute(bool): Recompute the column stats (default: False)
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

    def make_ready(self) -> bool:
        """This is a BLOCKING method that will wait until the FeatureSet is ready"""

        # Make sure the FeatureSet Details are computed
        self.details()

        # Call our underlying DataSource make_ready method
        if not self.data_source.make_ready():
            return False

        # Set ourselves to ready
        self.set_status("ready")
        self.refresh_meta()


if __name__ == "__main__":
    """Exercise for FeatureSet Class"""
    from pprint import pprint

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Grab a FeatureSet object and pull some information from it
    my_features = FeatureSet("test_feature_set")

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
    columns = my_features.column_names()
    print(columns)

    # Get the metadata and tags associated with this feature set
    print(f"SageWorks Meta: {my_features.sageworks_meta()}")
    print(f"SageWorks Tags: {my_features.sageworks_tags()}")

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
    outlier_df = my_features.outliers(recompute=True)
    print(outlier_df)

    # Now delete the AWS artifacts associated with this Feature Set
    # print('Deleting SageWorks Feature Set...')
    # my_features.delete()
