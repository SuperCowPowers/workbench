"""PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet (Athena/FeatureStore)"""
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import time
import botocore
from sagemaker.feature_store.feature_group import FeatureGroup

# Local imports
from sageworks.utils.iso_8601 import datetime_to_iso8601
from sageworks.transforms.transform import Transform
from sageworks.artifacts.feature_sets.feature_set import FeatureSet


class PandasToFeatures(Transform):
    """PandasToFeatures: Class to publish a Pandas DataFrame into a FeatureSet (Athena/FeatureStore)

    Common Usage:
        to_features = PandasToFeatures(output_uuid)
        to_features.set_output_tags(["abalone", "public", "whatever"])
        to_features.set_input(df, id_column="id"/None, event_time_column="date"/None)
        to_features.transform(delete_existing=True/False)
    """

    def __init__(self, output_uuid: str):
        """PandasToFeatures Initialization"""

        # Call superclass init
        super().__init__("DataFrame", output_uuid)

        # Set up all my instance attributes
        self.id_column = None
        self.event_time_column = None
        self.output_df = None

    def set_input(self, input_df: pd.DataFrame, id_column=None, event_time_column=None):
        """Set the Input DataFrame for this Transform"""
        self.id_column = id_column
        self.event_time_column = event_time_column
        self.output_df = input_df.copy()

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
            current_datetime = datetime.now(timezone.utc)
            self.log.info("Generating an event_time column before FeatureSet Creation...")
            self.event_time_column = "event_time"
            self.output_df[self.event_time_column] = pd.Series([current_datetime] * len(self.output_df))

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
    def categorical_converter(self):
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

    @staticmethod
    def convert_nullable_types(df: pd.DataFrame) -> pd.DataFrame:
        """Convert the new Pandas 'nullable types' since AWS SageMaker code doesn't currently support them
        See: https://github.com/aws/sagemaker-python-sdk/pull/3740"""
        for column in list(df.select_dtypes(include=[pd.Int64Dtype]).columns):
            df[column] = df[column].astype("int64")
        for column in list(df.select_dtypes(include=[pd.Float64Dtype]).columns):
            df[column] = df[column].astype("float64")
        return df

    def transform_impl(self, delete_existing=True):
        """Convert the Pandas DataFrame into Parquet Format in the SageWorks S3 Bucket, and
        store the information about the data to the AWS Data Catalog sageworks database"""

        # Ensure that the input dataframe has id and event_time fields
        self._ensure_id_column()
        self._ensure_event_time()

        # Convert object and string types to Categorical
        self.categorical_converter()

        # Convert Int64 and Float64 types (see: https://github.com/aws/sagemaker-python-sdk/pull/3740)
        self.output_df = self.convert_nullable_types(self.output_df)

        # FeatureSet Internal Storage (Athena) will convert columns names to lowercase, so we need
        # to make sure that the column names are lowercase to match and avoid downstream issues
        self.output_df.columns = self.output_df.columns.str.lower()

        # Mark 80% of the data as training and 20% as validation/test
        self.output_df["training"] = np.random.binomial(size=len(self.output_df), n=1, p=0.8)

        # Do we want to delete the existing FeatureSet?
        if delete_existing:
            try:
                delete_fs = FeatureSet(self.output_uuid)
                if delete_fs.check():
                    delete_fs.delete()
            except botocore.exceptions.ClientError as exc:
                self.log.info(f"FeatureSet {self.output_uuid} doesn't exist...")
                self.log.info(exc)

        # Create a Feature Group and load our Feature Definitions
        my_feature_group = FeatureGroup(name=self.output_uuid, sagemaker_session=self.sm_session)
        my_feature_group.load_feature_definitions(data_frame=self.output_df)

        # Create the Output Parquet file S3 Storage Path for this Feature Set
        s3_storage_path = f"{self.feature_sets_s3_path}/{self.output_uuid}"

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
        my_feature_group.ingest(self.output_df, max_processes=4)

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
    pd.set_option("display.max_colwidth", 15)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Create some fake data
    my_datetime = datetime.now(timezone.utc)
    fake_data = [
        {"id": 1, "name": "sue", "age": 41, "score": 7.8, "date": my_datetime},
        {"id": 2, "name": "bob", "age": 34, "score": 6.4, "date": my_datetime},
        {"id": 3, "name": "ted", "age": 69, "score": 8.2, "date": my_datetime},
        {"id": 4, "name": "bill", "age": 24, "score": 5.3, "date": my_datetime},
        {"id": 5, "name": "sally", "age": 52, "score": 9.5, "date": my_datetime},
    ]
    fake_df = pd.DataFrame(fake_data)

    # Create my DF to Feature Set Transform
    df_to_features = PandasToFeatures("test_feature_set")
    df_to_features.set_input(fake_df, id_column="id", event_time_column="date")
    df_to_features.set_output_tags(["test", "small"])

    # Store this dataframe as a SageWorks Feature Set
    df_to_features.transform()
