"""PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet (Athena/FeatureStore)"""
from datetime import datetime, timezone
import pandas as pd
import time
from sagemaker.feature_store.feature_group import FeatureGroup

# Local imports
from sageworks.transforms.transform import Transform
from sageworks.artifacts.feature_sets.feature_set import FeatureSet


# Good Reference for AWS FeatureStore functionality
# - https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-featurestore/feature_store_introduction.html
# AWS ROLES
# IMPORTANT: You must attach the following policies to your execution role: * AmazonS3FullAccess * AmazonSageMakerFeatureStoreAccess


class PandasToFeatures(Transform):
    def __init__(self):
        """PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet (Athena/FeatureStore)"""

        # Call superclass init
        super().__init__()

        # Set up all my instance attributes
        self.input_df = None
        self.id_column = None
        self.event_time_column = None

    def set_input(self, input_df: pd.DataFrame, id_column, event_time_column=None):
        """Set the Input DataFrame for this Transform"""
        self.id_column = id_column
        self.event_time_column = event_time_column
        self.input_df = input_df

    def _ensure_event_time(self):
        """Internal: AWS Feature Store requires an EventTime field in all data stored"""
        if self.event_time_column is None or self.event_time_column not in self.input_df.columns:
            current_datetime = datetime.now(timezone.utc)
            self.log.info('Generating an EventTime column before FeatureSet Creation...')
            self.event_time_column = 'EventTime'
            self.input_df[self.event_time_column] = pd.Series([current_datetime] * len(self.input_df))

        # The event_time_column is defined so lets make sure it the right type for Feature Store
        if pd.api.types.is_datetime64_any_dtype(self.input_df[self.event_time_column]):
            self.log.info(f"Converting {self.event_time_column} to ISOFormat Date String before FeatureSet Creation...")

            # Convert the datetime DType to ISO-8601 string
            self.input_df[self.event_time_column] = self.input_df[self.event_time_column].map(self._iso8601_utc)

    @staticmethod
    def _iso8601_utc(dt):
        """convert datetime to string in UTC format (yyyy-MM-dd'T'HH:mm:ss.SSSZ)"""
        iso_str = dt.astimezone(timezone.utc).isoformat('T', 'milliseconds')
        return iso_str.replace('+00:00', 'Z')

    def _convert_objs_to_string(self):
        """Internal: AWS Feature Store doesn't know how to store object dtypes, so convert to String"""
        for col in self.input_df:
            if pd.api.types.is_object_dtype(self.input_df[col].dtype):
                self.input_df[col] = self.input_df[col].astype(pd.StringDtype())

    def transform_impl(self, delete_existing=False):
        """Convert the Pandas DataFrame into Parquet Format in the SageWorks S3 Bucket, and
           store the information about the data to the AWS Data Catalog sageworks database"""

        # Ensure that the input dataframe has an EventTime field
        self._ensure_event_time()

        # Convert object dtypes to string
        self._convert_objs_to_string()

        # Do we want to delete the existing FeatureSet?
        if delete_existing:
            try:
                FeatureSet(self.output_uuid).delete()
            except TypeError:
                self.log.info(f"FeatureSet {self.output_uuid} doesn't exist...")

        # Create a Feature Group and load our Feature Definitions
        my_feature_group = FeatureGroup(name=self.output_uuid, sagemaker_session=self.sm_session)
        my_feature_group.load_feature_definitions(data_frame=self.input_df)

        # Add some tags here
        tags = ['sageworks', 'public']
        print(tags)

        # Create the Output Parquet file S3 Storage Path for this Feature Set
        s3_storage_path = f"{self.feature_set_s3_path}/{self.output_uuid}"

        # Data Catalog Config
        # FIXME: AWS wants to put Feature Groups into the 'sagemaker_features' database
        #        and a random table name. We can overwrite these but then the AWS Glue
        #        Catalog Table isn't automatically created.
        """
        my_config = DataCatalogConfig(table_name=self.output_uuid,
                                      catalog='AwsDataCatalog',
                                      database='sageworks')
        """

        # Write out the DataFrame to Parquet/FeatureSet/Athena
        my_feature_group.create(
            s3_uri=s3_storage_path,
            record_identifier_name=self.id_column,
            event_time_feature_name=self.event_time_column,
            role_arn=self.sageworks_role_arn,
            enable_online_store=True
            # data_catalog_config=my_config,
            # disable_glue_table_creation=True
        )

        # Ensure/wait for the feature group to be created
        self.ensure_feature_group_created(my_feature_group)

        # Now we actually push the data into the Feature Group
        my_feature_group.ingest(self.input_df, max_processes=4)

    def ensure_feature_group_created(self, feature_group):
        status = feature_group.describe().get("FeatureGroupStatus")
        while status == "Creating":
            self.log.info("FeatureSet being Created...")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
        self.log.info(f"FeatureSet {feature_group.name} successfully created")


# Simple test of the PandasToFeatures functionality
def test():
    """Test the PandasToFeatures Class"""

    # Setup Pandas output options
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    # Create some fake data
    my_datetime = datetime.now(timezone.utc)
    fake_data = [
        {'id': 1, 'name': 'sue', 'age': 41, 'score': 7.8, 'date': my_datetime},
        {'id': 2, 'name': 'bob', 'age': 34, 'score': 6.4, 'date': my_datetime},
        {'id': 3, 'name': 'ted', 'age': 69, 'score': 8.2, 'date': my_datetime},
        {'id': 4, 'name': 'bill', 'age': 24, 'score': 5.3, 'date': my_datetime},
        {'id': 5, 'name': 'sally', 'age': 52, 'score': 9.5, 'date': my_datetime}
        ]
    fake_df = pd.DataFrame(fake_data)

    # Create my DF to Feature Set Transform
    output_uuid = 'test-feature-set'
    df_to_features = PandasToFeatures()
    df_to_features.set_input(fake_df, id_column='id', event_time_column='date')
    df_to_features.set_output_uuid(output_uuid)

    # Store this dataframe as a SageWorks Feature Set
    df_to_features.transform(delete_existing=True)


if __name__ == "__main__":
    test()
