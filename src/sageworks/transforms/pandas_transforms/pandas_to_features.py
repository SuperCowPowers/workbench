"""PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet (Athena/FeatureStore)"""
from datetime import datetime, timezone
import pandas as pd
import time
import botocore
from sagemaker.feature_store.feature_group import FeatureGroup

# Local imports
from sageworks.transforms.transform import Transform
from sageworks.artifacts.feature_sets.feature_set import FeatureSet


# Good Reference for AWS FeatureStore functionality
# - https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-featurestore/feature_store_introduction.html
# AWS ROLES
# IMPORTANT: You must attach the following policies to your execution role: * AmazonS3FullAccess * AmazonSageMakerFeatureStoreAccess


class PandasToFeatures(Transform):
    def __init__(self, output_uuid):
        """PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet (Athena/FeatureStore)"""

        # Call superclass init
        super().__init__(None, output_uuid)

        # Set up all my instance attributes
        self.input_df = None
        self.id_column = None
        self.event_time_column = None

    def set_input(self, input_df: pd.DataFrame, id_column=None, event_time_column=None):
        """Set the Input DataFrame for this Transform"""
        self.id_column = id_column
        self.event_time_column = event_time_column
        self.input_df = input_df

    def _ensure_id_column(self):
        """Internal: AWS Feature Store requires an Id field for all data store"""
        if self.id_column is None or self.id_column not in self.input_df.columns:
            if 'id' not in self.input_df.columns:
                self.log.info('Generating an id column before FeatureSet Creation...')
                self.input_df['id'] = self.input_df.index
            self.id_column = 'id'

    def _ensure_event_time(self):
        """Internal: AWS Feature Store requires an event_time field for all data stored"""
        if self.event_time_column is None or self.event_time_column not in self.input_df.columns:
            current_datetime = datetime.now(timezone.utc)
            self.log.info('Generating an event_time column before FeatureSet Creation...')
            self.event_time_column = 'event_time'
            self.input_df[self.event_time_column] = pd.Series([current_datetime] * len(self.input_df))

        # The event_time_column is defined so lets make sure it the right type for Feature Store
        if pd.api.types.is_datetime64_any_dtype(self.input_df[self.event_time_column]):
            self.log.info(f"Converting {self.event_time_column} to ISOFormat Date String before FeatureSet Creation...")

            # Convert the datetime DType to ISO-8601 string
            self.input_df[self.event_time_column] = self.input_df[self.event_time_column].map(self._iso8601_utc)
            self.input_df[self.event_time_column] = self.input_df[self.event_time_column].astype(pd.StringDtype())

    @staticmethod
    def _iso8601_utc(dt):
        """convert datetime to string in UTC format (yyyy-MM-dd'T'HH:mm:ss.SSSZ)"""

        # Check for TimeZone
        if dt.tzinfo is None:
            dt = dt.tz_localize(timezone.utc)

        # Convert to ISO-8601 String
        iso_str = dt.astimezone(timezone.utc).isoformat('T', 'milliseconds')
        return iso_str.replace('+00:00', 'Z')

    def _convert_objs_to_string(self):
        """Internal: AWS Feature Store doesn't know how to store object dtypes, so convert to String"""
        for col in self.input_df:
            if pd.api.types.is_object_dtype(self.input_df[col].dtype):
                self.input_df[col] = self.input_df[col].astype(pd.StringDtype())

    # Helper Methods
    def categorical_converter(self):
        """Convert object and string types to Categorical"""
        categorical_columns = []
        for feature, dtype in self.input_df.dtypes.items():
            print(feature, dtype)
            if dtype in ['object', 'string'] and feature not in [self.event_time_column, self.id_column]:
                print(f"Converting object column {feature} to categorical")
                print(f"Unique Values = {self.input_df[feature].nunique()}")
                self.input_df[feature] = self.input_df[feature].astype("category")
                categorical_columns.append(feature)

        # Now convert Categorical Types to One Hot Encoding
        self.input_df = pd.get_dummies(self.input_df, columns=categorical_columns)

    @staticmethod
    def convert_nullable_types(df: pd.DataFrame) -> pd.DataFrame:
        """Convert the new Pandas 'nullable types' since AWS SageMaker code doesn't currently support them
           See: https://github.com/aws/sagemaker-python-sdk/pull/3740"""
        for column in list(df.select_dtypes(include=[pd.Int64Dtype]).columns):
            df[column] = df[column].astype('int64')
        for column in list(df.select_dtypes(include=[pd.Float64Dtype]).columns):
            df[column] = df[column].astype('float64')
        return df

    def transform_impl(self, delete_existing=False):
        """Convert the Pandas DataFrame into Parquet Format in the SageWorks S3 Bucket, and
           store the information about the data to the AWS Data Catalog sageworks database"""

        # Ensure that the input dataframe has id and event_time fields
        self._ensure_id_column()
        self._ensure_event_time()

        # Convert object and string types to Categorical
        self.categorical_converter()

        # Convert Int64 and Float64 types (see: https://github.com/aws/sagemaker-python-sdk/pull/3740)
        self.input_df = self.convert_nullable_types(self.input_df)

        # Do we want to delete the existing FeatureSet?
        if delete_existing:
            try:
                FeatureSet(self.output_uuid).delete()
            except botocore.exceptions.ClientError:
                self.log.info(f"FeatureSet {self.output_uuid} doesn't exist...")

        # Create a Feature Group and load our Feature Definitions
        my_feature_group = FeatureGroup(name=self.output_uuid, sagemaker_session=self.sm_session)
        my_feature_group.load_feature_definitions(data_frame=self.input_df)

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

        # Get the metadata/tags to push into AWS
        aws_tags = self.get_aws_tags()

        # Write out the DataFrame to Parquet/FeatureSet/Athena
        my_feature_group.create(
            s3_uri=s3_storage_path,
            record_identifier_name=self.id_column,
            event_time_feature_name=self.event_time_column,
            role_arn=self.sageworks_role_arn,
            enable_online_store=True,
            tags=aws_tags
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


if __name__ == "__main__":
    """Exercise the PandasToFeatures Class"""

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
    df_to_features = PandasToFeatures('test_feature_set')
    df_to_features.set_input(fake_df, id_column='id', event_time_column='date')
    df_to_features.set_output_tags(['test', 'small'])
    df_to_features.set_output_meta({'sageworks_input': 'DataFrame'})

    # Store this dataframe as a SageWorks Feature Set
    df_to_features.transform(delete_existing=True)
