"""FeatureSet: SageWork Feature Set accessible through Athena"""
import time
import logging
from sagemaker.feature_store.feature_group import FeatureGroup

# SageWorks Imports
from sageworks.artifacts.data_sources.athena_source import AthenaSource
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory, AWSServiceBroker
from sageworks.aws_service_broker.aws_sageworks_role_manager import AWSSageWorksRoleManager
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()


class FeatureSet(AthenaSource):

    def __init__(self, feature_set_name):
        """FeatureSet: SageWork Feature Set accessible through Athena

        Args:
            feature_set_name (str): Name of Feature Set in SageWorks Metadata.
        """
        self.log = logging.getLogger(__name__)
        self.feature_set_name = feature_set_name

        # Grab an AWS Metadata Broker object and pull information for Feature Sets
        self.aws_meta = AWSServiceBroker()
        self.feature_meta = self.aws_meta.get_metadata(ServiceCategory.FEATURE_STORE).get(self.feature_set_name)

        # Pull Athena and S3 Storage information from metadata
        self.athena_database = self.feature_meta['sageworks'].get('athena_database')
        self.athena_table = self.feature_meta['sageworks'].get('athena_table')
        self.s3_storage = self.feature_meta['sageworks'].get('s3_storage')

        # Grab our SageMaker Session in case we need it later
        self.sm_session = AWSSageWorksRoleManager().sagemaker_session()

        # Call SuperClass (AthenaSource) Initialization
        super().__init__(self.athena_table, self.athena_database)

        # All done
        self.log.info(f"FeatureSet Initialized: {feature_set_name}")

    def check(self) -> bool:
        """Does the feature_set_name exist in the AWS Metadata?"""
        if self.feature_meta is None:
            self.log.critical(f'FeatureSet.check() {self.feature_set_name} not found in AWS Metadata!')
            return False
        return True

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

    # Grab a FeatureSet object and pull some information from it
    my_features = FeatureSet('test-feature-set')

    # Call the various methods

    # Lets do a check/validation of the feature set
    assert(my_features.check())

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


if __name__ == "__main__":
    test()
