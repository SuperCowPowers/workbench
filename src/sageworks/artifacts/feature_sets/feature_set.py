"""FeatureSet: SageWorks Feature Set accessible through Athena"""
import time
from datetime import datetime, timezone

import botocore.exceptions
import pandas as pd
import awswrangler as wr

from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_store import FeatureStore

# SageWorks Imports
from sageworks.artifacts.artifact import Artifact
from sageworks.artifacts.data_sources.athena_source import AthenaSource
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory


class FeatureSet(Artifact):
    """FeatureSet: SageWorks Feature Set accessible through Athena

    Common Usage:
        my_features = FeatureSet(feature_uuid)
        my_features.summary()
        my_features.details()
    """

    def __init__(self, feature_uuid):
        """FeatureSet Initialization

        Args:
            feature_uuid (str): Name of Feature Set in SageWorks Metadata.
        """
        # Call superclass init
        super().__init__(feature_uuid)

        # Grab an AWS Metadata Broker object and pull information for Feature Sets
        self.feature_set_name = feature_uuid
        self.feature_meta = self.aws_meta.get_metadata(ServiceCategory.FEATURE_STORE).get(self.feature_set_name)
        if self.feature_meta is None:
            self.log.warning(f"Could not find feature set {self.feature_set_name} within current visibility scope")
            self.athena_source = None
            return
        else:
            self.record_id = self.feature_meta["RecordIdentifierFeatureName"]
            self.event_time = self.feature_meta["EventTimeFeatureName"]

            # Pull Athena and S3 Storage information from metadata
            self.athena_database = self.feature_meta["sageworks"].get("athena_database")
            self.athena_table = self.feature_meta["sageworks"].get("athena_table")
            self.s3_storage = self.feature_meta["sageworks"].get("s3_storage")

            # Creat our internal AthenaSource
            self.athena_source = AthenaSource(self.athena_table, self.athena_database)

        # Spin up our Feature Store
        self.feature_store = FeatureStore(self.sm_session)

        # All done
        self.log.info(f"FeatureSet Initialized: {self.feature_set_name}")

    def check(self) -> bool:
        """Does the feature_set_name exist in the AWS Metadata?"""
        if self.feature_meta is None:
            self.log.critical(f"FeatureSet.check() {self.feature_set_name} not found in AWS Metadata!")
            return False
        return True

    def meta(self) -> dict:
        """Get the full AWS metadata for this artifact"""
        return self.feature_meta

    def arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for this artifact"""
        return self.feature_meta["FeatureGroupArn"]

    def size(self) -> float:
        """Return the size of the internal AthenaSource in MegaBytes"""
        return self.athena_source.size()

    def column_names(self) -> list[str]:
        """Return the column names of the internal AthenaSource"""
        return self.athena_source.column_names()

    def column_types(self) -> list[str]:
        """Return the column types of the internal AthenaSource"""
        return self.athena_source.column_types()

    def column_details(self) -> list[str]:
        """Return the column types of the internal AthenaSource"""
        return self.athena_source.column_details()

    def num_columns(self) -> int:
        """Return the number of columns of the internal AthenaSource"""
        return len(self.column_names())

    def num_rows(self) -> int:
        """Return the number of rows of the internal AthenaSource"""
        return self.athena_source.num_rows()

    def query(self, query: str) -> pd.DataFrame:
        """Return the number of rows of the internal AthenaSource"""
        return self.athena_source.query(query)

    def aws_url(self):
        """The AWS URL for looking at/querying this data source"""
        return "https://us-west-2.console.aws.amazon.com/athena/home"

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self.feature_meta["CreationTime"]

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        # Note: We can't currently figure out how to this from AWS Metadata
        return self.feature_meta["CreationTime"]

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
        s3_output_path = self.feature_sets_s3_path + f"/{self.feature_set_name}/datasets/all_{date_time}"

        # Get the snapshot query
        query = self.snapshot_query()

        # Make the query
        athena_query = FeatureGroup(name=self.feature_set_name, sagemaker_session=self.sm_session).athena_query()
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

    def details(self) -> dict:
        """Additional Details about this FeatureSet"""
        details = self.summary()
        # Fill in more details here if needed
        return details

    def delete(self):
        """Delete the Feature Set: Feature Group, Catalog Table, and S3 Storage Objects"""

        # Delete the Feature Group and ensure that it gets deleted
        remove_fg = FeatureGroup(name=self.feature_set_name, sagemaker_session=self.sm_session)
        remove_fg.delete()
        self.ensure_feature_group_deleted(remove_fg)

        # Delete our underlying AthenaSource (Data Catalog Table and S3 Storage Objects)
        self.athena_source.delete()

        # Feature Sets can often have a lot of cruft so delete the entire bucket/prefix
        s3_delete_path = self.feature_sets_s3_path + f"/{self.feature_set_name}"
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


# Simple test of the FeatureSet functionality
def test():
    """Test for FeatureSet Class"""

    # Grab a FeatureSet object and pull some information from it
    my_features = FeatureSet("test_feature_set")

    # Call the various methods

    # Let's do a check/validation of the feature set
    print(f"Feature Set Check: {my_features.check()}")

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

    # Now delete the AWS artifacts associated with this Feature Set
    # print('Deleting SageWorks Feature Set...')
    # my_features.delete()


if __name__ == "__main__":
    test()
