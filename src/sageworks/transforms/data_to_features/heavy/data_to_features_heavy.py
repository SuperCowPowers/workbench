"""DataToFeaturesHeavy: Class to Transform a DataSource into a FeatureSet (Athena/FeatureStore)"""
import pandas as pd
import time
import botocore
from sagemaker.feature_store.feature_group import FeatureGroup
import awswrangler as wr
from sagemaker.feature_store.inputs import DataCatalogConfig

# Local imports
from sageworks.transforms.transform import Transform
from sageworks.artifacts.data_sources.data_source import DataSource
from sageworks.artifacts.feature_sets.feature_set import FeatureSet


class DataToFeaturesHeavy(Transform):
    """DataToFeaturesHeavy: Class to Transform a DataSource into a FeatureSet (Athena/FeatureStore)

    Common Usage:
        to_features = DataToFeaturesHeavy(output_uuid)
        to_features.set_output_tags(["abalone", "heavy", "whatever"])
        to_features.set_input(df, id_column="id"/None, event_time_column="date"/None)
        to_features.transform(delete_existing=True/False)
    """

    def __init__(self, input_uuid: str, output_uuid: str):
        """DataToFeaturesHeavy Initialization"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.id_column = None
        self.event_time_column = None
        self.input_data_source = DataSource(input_uuid)
        self.input_sample_df = self.input_data_source.sample_df()
        self.output_database = "sagemaker_featurestore"

    @staticmethod
    def convert_nullable_types(df: pd.DataFrame) -> pd.DataFrame:
        """Convert the new Pandas 'nullable types' since AWS SageMaker code doesn't currently support them
        See: https://github.com/aws/sagemaker-python-sdk/pull/3740"""
        for column in list(df.select_dtypes(include=[pd.Int32Dtype]).columns):
            df[column] = df[column].astype("int32")
        for column in list(df.select_dtypes(include=[pd.Int64Dtype]).columns):
            df[column] = df[column].astype("int64")
        for column in list(df.select_dtypes(include=[pd.Float64Dtype]).columns):
            df[column] = df[column].astype("float64")
        return df

    def transform_impl(self, id_column: str, event_time_column: str = None, delete_existing: bool = True):
        """Convert the Data Source into a Feature Set, also storing the information
        about the data to the AWS Data Catalog sageworks database and create S3 Objects"""

        # Do we want to delete the existing FeatureSet?
        if delete_existing:
            try:
                delete_fs = FeatureSet(self.output_uuid)
                if delete_fs.check():
                    delete_fs.delete()
            except botocore.exceptions.ClientError as exc:
                self.log.info(f"FeatureSet {self.output_uuid} doesn't exist...")
                self.log.info(exc)

        # Set the ID and Event Time Columns
        self.id_column = id_column
        self.event_time_column = event_time_column

        # Create the Output Parquet file S3 Storage Path for this Feature Set
        s3_storage_path = f"{self.feature_sets_s3_path}"

        # Now we'll make a Query of the Data Source to create a FeatureSet
        query = f"SELECT * FROM {self.input_data_source.table_name}"
        self.log.info(f"Creating FeatureSet Data with Query: {query}")
        info = wr.athena.create_ctas_table(
            sql=query,
            database=self.input_data_source.data_catalog_db,
            ctas_table=self.output_uuid,
            ctas_database=self.output_database,
            s3_output=s3_storage_path,
            write_compression="snappy",
            boto3_session=self.boto_session,
            wait=True
        )
        self.log.info(f"FeatureSet Data Created: {info}")

        # Convert Int32, Int64 and Float64 types (see: https://github.com/aws/sagemaker-python-sdk/pull/3740)
        self.input_sample_df = self.convert_nullable_types(self.input_sample_df)

        # Create a Feature Group and load our Feature Definitions
        self.log.info(f"Creating FeatureGroup: {self.output_uuid}")
        my_feature_group = FeatureGroup(name=self.output_uuid, sagemaker_session=self.sm_session)
        my_feature_group.load_feature_definitions(data_frame=self.input_sample_df)

        # Get the metadata/tags to push into AWS
        aws_tags = self.get_aws_tags()

        # Data Catalog Config
        my_config = DataCatalogConfig(table_name=self.output_uuid,
                                      catalog='AwsDataCatalog',
                                      database=self.output_database)

        # Write out the DataFrame to Parquet/FeatureSet/Athena
        my_feature_group.create(
            s3_uri=s3_storage_path,
            record_identifier_name=self.id_column,
            event_time_feature_name=self.event_time_column,
            role_arn=self.sageworks_role_arn,
            enable_online_store=True,
            tags=aws_tags,
            data_catalog_config=my_config,
            disable_glue_table_creation=True
        )

        # Ensure/wait for the feature group to be created
        self.ensure_feature_group_created(my_feature_group)

    def ensure_feature_group_created(self, feature_group):
        status = feature_group.describe().get("FeatureGroupStatus")
        while status == "Creating":
            self.log.info("FeatureSet being Created...")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
        self.log.info(f"FeatureSet {feature_group.name} successfully created")


if __name__ == "__main__":
    """Exercise the DataToFeaturesHeavy Class"""

    # Create my DF to Feature Set Transform
    data_to_features_heavy = DataToFeaturesHeavy("heavy_data_test", "heavy_data_test_features")
    data_to_features_heavy.set_output_tags(["test", "heavy"])

    # Store this dataframe as a SageWorks Feature Set
    data_to_features_heavy.transform(id_column="id", event_time_column="date")
