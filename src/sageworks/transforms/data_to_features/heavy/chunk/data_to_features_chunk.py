"""DataToFeaturesChunk: Class to Transform a DataSource into a FeatureSet using Chunking"""
import time
import botocore
import pandas as pd
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.inputs import TableFormatEnum

# Local imports
from sageworks.transforms.transform import Transform
from sageworks.artifacts.data_sources.data_source import DataSource
from sageworks.artifacts.feature_sets.feature_set import FeatureSet


class DataToFeaturesChunk(Transform):
    """DataToFeaturesChunk: Class to Transform a DataSource into a FeatureSet using Chunking

    Common Usage:
        to_features = DataToFeaturesChunk(output_uuid)
        to_features.set_output_tags(["heavy", "whatever"])
        to_features.set_chunk_size(10000)  # Default is 5000 can also be None (no chunking)
        to_features.transform(query, id_column, event_time_column=None)
    """

    def __init__(self, input_uuid: str, output_uuid: str):
        """DataToFeaturesChunk Initialization"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.id_column = None
        self.event_time_column = None
        self.input_data_source = DataSource(input_uuid)
        self.output_database = "sagemaker_featurestore"
        self.table_format = TableFormatEnum.ICEBERG

    @staticmethod
    def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
        """Convert the types of the DataFrame to the correct types for the Feature Store"""
        datetime_type = ["datetime", "datetime64", "datetime64[ns]", "datetimetz"]
        for column in df.select_dtypes(include=datetime_type).columns:
            df[column] = df[column].astype("string")
        for column in list(df.select_dtypes(include=[pd.Int64Dtype]).columns):
            df[column] = df[column].astype("int64")
        for column in list(df.select_dtypes(include=[pd.Float64Dtype]).columns):
            df[column] = df[column].astype("float64")
        return df

    def transform_impl(self, query, id_column: str, event_time_column: str = None):
        """Convert the Data Source into a Feature Set, also storing the information
        about the data to the AWS Data Catalog sageworks database and create S3 Objects"""

        # Do we want to delete the existing FeatureSet?
        try:
            delete_fs = FeatureSet(self.output_uuid)
            if delete_fs.check():
                delete_fs.delete()
                time.sleep(5)
        except botocore.exceptions.ClientError as exc:
            self.log.info(f"FeatureSet {self.output_uuid} doesn't exist...")
            self.log.info(exc)

        # Set the ID and Event Time Columns
        self.id_column = id_column
        self.event_time_column = event_time_column

        # Now we're going to use the Athena Query to create a sample DataFrame
        sample_df = self.input_data_source.query(query + " LIMIT 1000")

        # We need to convert some of our column types to the correct types
        # Feature Store only supports these data types:
        # - Integral
        # - Fractional
        # - String (timestamp/datetime types need to be converted to string)
        sample_df = self.convert_column_types(sample_df)

        # Create a Feature Group and load our Feature Definitions
        self.log.info(f"Creating FeatureGroup: {self.output_uuid}")
        my_feature_group = FeatureGroup(name=self.output_uuid, sagemaker_session=self.sm_session)
        my_feature_group.load_feature_definitions(data_frame=sample_df)

        # Get the metadata/tags to push into AWS
        aws_tags = self.get_aws_tags()

        # Write out the DataFrame to Parquet/FeatureSet/Athena

        # Create the Feature Group
        my_feature_group.create(
            s3_uri=self.feature_sets_s3_path,
            record_identifier_name=self.id_column,
            event_time_feature_name=self.event_time_column,
            role_arn=self.sageworks_role_arn,
            enable_online_store=True,
            table_format=self.table_format,
            tags=aws_tags,
        )

        # Ensure/wait for the feature group to be created
        self.ensure_feature_group_created(my_feature_group)

        # Now we're going to use chunks to ingest the data into the Feature Group
        self.log.info(f"Ingesting data into feature group: {my_feature_group.name} ...")
        my_feature_group.ingest(data_frame=sample_df, max_processes=16, wait=True)
        self.log.info(f"{len(sample_df)} records ingested into feature group: {my_feature_group.name}")

        # Create the FeatureSet Object
        self.log.info(f"Creating FeatureSet Object: {self.output_uuid}")
        my_feature_set = FeatureSet(self.output_uuid, force_refresh=True)

        # Compute Details, Quartiles, and SampleDF from the Feature Group
        my_feature_set.details()
        my_feature_set.data_source.details()
        my_feature_set.quartiles()
        my_feature_set.sample_df()
        my_feature_set.set_status("ready")
        self.log.info("FeatureSet Object Created")

    def ensure_feature_group_created(self, feature_group):
        status = feature_group.describe().get("FeatureGroupStatus")
        while status == "Creating":
            self.log.info("FeatureSet being Created...")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
        self.log.info(f"FeatureSet {feature_group.name} successfully created")
        time.sleep(5)


if __name__ == "__main__":
    """Exercise the DataToFeaturesChunk Class"""

    # Create my DF to Feature Set Transform
    data_to_features_heavy = DataToFeaturesChunk("heavy_dns", "dns_features_test")
    data_to_features_heavy.set_output_tags(["test", "heavy"])

    # Store this dataframe as a SageWorks Feature Set
    fields = ["timestamp", "flow_id", "in_iface", "proto", "dns_type", "dns_rrtype", "dns_flags", "dns_rcode"]
    query = f"SELECT {', '.join(fields)} FROM heavy_dns"
    data_to_features_heavy.transform(query=query, id_column="flow_id", event_time_column="timestamp")
