"""DataSourceFactory: A Factory for DataSources (Athena, RDS, etc)"""

# Local imports
from workbench.core.artifacts.athena_source import AthenaSource


class DataSourceFactory:
    """DataSourceFactory: Workbench DataSource is the best source for your data"""

    def __new__(cls, uuid, data_source_type: str = "athena", force_refresh: bool = False):
        """DataSourceFactory: A Factory for DataSources (Athena, RDS, etc)
        Args:
            uuid: The UUID of the DataSource
            data_source_type: The type of DataSource (athena, rds, etc)
            force_refresh: Force a refresh of the AWS Broker (default: False)
        Returns:
            object: A concrete DataSource class (AthenaSource, RDSSource)
        """
        if data_source_type == "athena":
            # We're going to check both regular DataSources and DataSources
            # that are storage locations for FeatureSets
            ds = AthenaSource(uuid)
            if ds.exists():
                return ds
            else:
                return AthenaSource(uuid, "sagemaker_featurestore")
        else:
            raise NotImplementedError(f"DataSource type {data_source_type} not implemented")


if __name__ == "__main__":
    """Exercise the DataSourceFactory Class"""
    from pprint import pprint

    # Retrieve a DataSourceFactory
    my_data = DataSourceFactory("abalone_data")

    # Verify that the Athena DataSource exists
    assert my_data.exists()

    # What's my Workbench UUID
    print(f"UUID: {my_data.uuid}")

    # What's my AWS ARN and URL
    print(f"AWS ARN: {my_data.arn()}")
    print(f"AWS URL: {my_data.aws_url()}")

    # Get the S3 Storage for this DataSource
    print(f"S3 Storage: {my_data.s3_storage_location()}")

    # What's the size of the data?
    print(f"Size of Data (MB): {my_data.size()}")

    # When was it created and last modified?
    print(f"Created: {my_data.created()}")
    print(f"Modified: {my_data.modified()}")

    # Column Names and Types
    print(f"Column Names: {my_data.columns}")
    print(f"Column Types: {my_data.column_types}")

    # Get Tags associated with this Artifact
    print(f"Tags: {my_data.get_tags()}")

    # Get ALL the AWS Metadata associated with this Artifact
    print("\n\nALL Meta")
    pprint(my_data.aws_meta())

    # Get a SAMPLE of the data
    print(f"Sample Data: {my_data.sample()}")
