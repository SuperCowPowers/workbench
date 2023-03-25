"""FeatureSet: SageWorks Feature Set accessible through Athena"""
import time
from datetime import datetime, timezone

from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_store import FeatureStore

# SageWorks Imports
from sageworks.artifacts.data_sources.athena_source import AthenaSource
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory, AWSServiceBroker
from sageworks.aws_service_broker.aws_sageworks_role_manager import AWSSageWorksRoleManager


class FeatureSet(AthenaSource):

    def __init__(self, feature_set_name):
        """FeatureSet: SageWorks Feature Set accessible through Athena

        Args:
            feature_set_name (str): Name of Feature Set in SageWorks Metadata.
        """

        # Grab an AWS Metadata Broker object and pull information for Feature Sets
        self.feature_set_name = feature_set_name
        self.aws_meta = AWSServiceBroker()
        self.feature_meta = self.aws_meta.get_metadata(ServiceCategory.FEATURE_STORE).get(self.feature_set_name)
        self.record_id = self.feature_meta['RecordIdentifierFeatureName']
        self.event_time = self.feature_meta['EventTimeFeatureName']

        # Pull Athena and S3 Storage information from metadata
        self.athena_database = self.feature_meta['sageworks'].get('athena_database')
        self.athena_table = self.feature_meta['sageworks'].get('athena_table')
        self.s3_storage = self.feature_meta['sageworks'].get('s3_storage')

        # Call SuperClass (AthenaSource) Initialization
        # Note: We would normally do this first, but we need the Athena table/db info
        super().__init__(self.athena_table, self.athena_database)

        # Grab our SageMaker Session
        self.sm_session = AWSSageWorksRoleManager().sagemaker_session()

        # Spin up our Feature Store
        self.feature_store = FeatureStore(self.sm_session)

        # All done
        self.log.info(f"FeatureSet Initialized: {feature_set_name}")

    def check(self) -> bool:
        """Does the feature_set_name exist in the AWS Metadata?"""
        if self.feature_meta is None:
            self.log.critical(f'FeatureSet.check() {self.feature_set_name} not found in AWS Metadata!')
            return False
        return True

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
        self.log.info('Waiting for Athena Query...')
        athena_query.wait()
        query_execution = athena_query.get_query_execution()

        # Get the full path to the S3 files with the results
        full_s3_path = s3_output_path + f"/{query_execution['QueryExecution']['QueryExecutionId']}.csv"
        return full_s3_path

    def snapshot_query(self):
        """An Athena query to get the latest snapshot of features"""
        # Remove FeatureGroup metadata columns that might have gotten added
        columns = self.column_names()
        filter_columns = ['write_time', 'api_invocation_time', 'is_deleted']
        columns = ', '.join([x for x in columns if x not in filter_columns])

        query = (f'SELECT {columns} '
                 f'    FROM (SELECT *, row_number() OVER (PARTITION BY {self.record_id} '
                 f'        ORDER BY {self.event_time} desc, api_invocation_time DESC, write_time DESC) AS row_num '
                 f'        FROM "{self.athena_table}") '
                 '    WHERE row_num = 1 and  NOT is_deleted;')
        return query

    def delete(self):
        """Delete the Feature Set: Feature Group, Catalog Table, and S3 Storage Objects"""

        # Delete the Feature Group and ensure that it gets deleted
        remove_fg = FeatureGroup(name=self.feature_set_name, sagemaker_session=self.sm_session)
        remove_fg.delete()
        self.ensure_feature_group_deleted(remove_fg)

        # Delete the Data Catalog Table and S3 Storage Objects
        super().delete()

    def ensure_feature_group_deleted(self, feature_group):
        status = feature_group.describe().get("FeatureGroupStatus")
        while status == "Deleting":
            self.log.info("FeatureSet being Deleted...")
            time.sleep(5)
            try:
                status = feature_group.describe().get("FeatureGroupStatus")
            except Exception:  # FIXME
                break
        self.log.info(f"FeatureSet {feature_group.name} successfully deleted")


# Simple test of the FeatureSet functionality
def test():
    """Test for FeatureSet Class"""
    from sageworks.transforms.pandas_transforms.features_to_pandas import FeaturesToPandas

    # Grab a FeatureSet object and pull some information from it
    my_features = FeatureSet('test-feature-set')

    # Call the various methods

    # Let's do a check/validation of the feature set
    assert (my_features.check())

    # How many rows and columns?
    num_rows = my_features.num_rows()
    num_columns = my_features.num_columns()
    print(f'Rows: {num_rows} Columns: {num_columns}')

    # What are the column names?
    columns = my_features.column_names()
    print(columns)

    # Get the tags associated with this feature set
    print(f"Tags: {my_features.tags()}")

    # Run a query to only pull back a few columns and rows
    column_query = ', '.join(columns[:3])
    query = f'select {column_query} from "{my_features.athena_database}"."{my_features.athena_table}" limit 10'
    df = my_features.query(query)
    print(df)

    # Create some training data from our Feature Set
    s3_path = my_features.create_s3_training_data()
    print('Training Data Created')
    print(s3_path)

    # Getting the Feature Set as a Pandas DataFrame
    my_df = FeaturesToPandas('test-feature-set').get_output()
    print('Feature Set as Pandas DataFrame')
    print(my_df.head())


if __name__ == "__main__":
    test()
