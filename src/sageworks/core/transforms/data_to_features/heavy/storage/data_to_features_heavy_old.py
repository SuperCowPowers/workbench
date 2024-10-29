"""DataToFeaturesHeavy: Class to Transform a DataSource into a FeatureSet (Athena/FeatureStore)"""

import time
import pandas as pd
from sagemaker.feature_store.feature_group import FeatureGroup
import awswrangler as wr
from sagemaker.feature_store.inputs import DataCatalogConfig

# Local imports
from sageworks.core.transforms.transform import Transform
from sageworks.core.artifacts.data_source_factory import DataSourceFactory
from sageworks.core.artifacts.feature_set_core import FeatureSetCore


class DataToFeaturesHeavy(Transform):
    """DataToFeaturesHeavy: Class to Transform a DataSource into a FeatureSet (Athena/FeatureStore)

    Common Usage:
        ```python`
        to_features = DataToFeaturesHeavy(output_uuid)
        to_features.set_output_tags(["heavy", "whatever"])
        to_features.transform(query, id_column, event_time_column=None)
        ```
    """

    def __init__(self, input_uuid: str, output_uuid: str):
        """DataToFeaturesHeavy Initialization"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Set up all my instance attributes
        self.id_column = None
        self.event_time_column = None
        self.input_data_source = DataSourceFactory(input_uuid)
        self.output_database = "sagemaker_featurestore"

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
        about the data to the AWS Data Catalog sageworks database and create S3 Objects
        """

        # Delete the existing FeatureSet (if it exists)
        FeatureSetCore.managed_delete(self.output_uuid)
        time.sleep(5)

        # Set the ID and Event Time Columns
        self.id_column = id_column
        self.event_time_column = event_time_column

        # Create the Output Parquet file S3 Storage Path for this Feature Set
        s3_storage_path = f"{self.feature_sets_s3_path}"

        # Now we'll make a Query of the Data Source to create a FeatureSet
        self.log.info(f"Creating FeatureSet Data with Query: {query}")
        info = wr.athena.create_ctas_table(
            sql=query,
            database=self.input_data_source.data_catalog_db,
            ctas_table=self.output_uuid,
            ctas_database=self.output_database,
            s3_output=s3_storage_path,
            write_compression="snappy",
            boto3_session=self.boto3_session,
            wait=True,
        )
        self.log.info(f"FeatureSet Data Created: {info}")

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

        # Data Catalog Config
        my_config = DataCatalogConfig(
            table_name=self.output_uuid,
            catalog="AwsDataCatalog",
            database=self.output_database,
        )

        # Write out the DataFrame to Parquet/FeatureSet/Athena
        my_feature_group.create(
            s3_uri=s3_storage_path,
            record_identifier_name=self.id_column,
            event_time_feature_name=self.event_time_column,
            role_arn=self.sageworks_role_arn,
            enable_online_store=True,
            tags=aws_tags,
            data_catalog_config=my_config,
            disable_glue_table_creation=True,
        )

        # Ensure/wait for the feature group to be created
        self.ensure_feature_group_created(my_feature_group)

        # Create the FeatureSet Object
        self.log.info(f"Creating FeatureSet Object: {self.output_uuid}")
        my_feature_set = FeatureSetCore(self.output_uuid)

        # Compute Details, Descriptive Stats, and SampleDF from the Feature Group
        my_feature_set.details()
        my_feature_set.data_source.details()
        my_feature_set.descriptive_stats()
        my_feature_set.sample()
        my_feature_set.set_status("ready")
        self.log.info("FeatureSet Object Created")

    def ensure_feature_group_created(self, feature_group):
        status = feature_group.describe().get("FeatureGroupStatus")
        while status == "Creating":
            self.log.debug("FeatureSet being Created...")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
        self.log.info(f"FeatureSet {feature_group.name} successfully created")
        time.sleep(5)


if __name__ == "__main__":
    """Exercise the DataToFeaturesHeavy Class"""

    # Create my DF to Feature Set Transform
    data_to_features_heavy = DataToFeaturesHeavy("heavy_dns", "dns_features_1")
    data_to_features_heavy.set_output_tags(["test", "heavy"])

    # Store this dataframe as a SageWorks Feature Set
    fields = [
        "timestamp",
        "flow_id",
        "in_iface",
        "proto",
        "dns_type",
        "dns_rrtype",
        "dns_flags",
        "dns_rcode",
    ]
    query = f"SELECT {', '.join(fields)} FROM heavy_dns"
    data_to_features_heavy.transform(query=query, id_column="flow_id", event_time_column="timestamp")
