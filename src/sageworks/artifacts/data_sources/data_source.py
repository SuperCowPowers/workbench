"""DataSource: A Factory for Data Sources (Athena, RDS, etc)"""

# Local imports
from sageworks.artifacts.data_sources.athena_source import AthenaSource


class DataSource:
    def __new__(cls, uuid, data_source_type: str = "athena"):
        """DataSource: A Factory for Data Sources (Athena, RDS, etc)
        Args:
            uuid: The UUID of the DataSource
            data_source_type: The type of DataSource (athena, rds, etc)
        Returns:
            object: A concrete DataSource class (AthenaSource, RDSSource)
        """
        if data_source_type == "athena":
            return AthenaSource(uuid)
        else:
            raise NotImplementedError(f"DataSource type {data_source_type} not implemented")


if __name__ == "__main__":
    """Exercise the DataSource Factory Class"""

    # Retrieve a Data Source
    my_data = DataSource("test_data")

    # Verify that the Athena Data Source exists
    assert my_data.check()

    # What's my SageWorks UUID
    print(f"UUID: {my_data.uuid}")

    # What's my AWS ARN
    print(f"AWS ARN: {my_data.arn()}")

    # Get the S3 Storage for this Data Source
    print(f"S3 Storage: {my_data.s3_storage_location()}")

    # What's the size of the data?
    print(f"Size of Data (MB): {my_data.size()}")

    # When was it created and last modified?
    print(f"Created: {my_data.created()}")
    print(f"Modified: {my_data.modified()}")

    # Column Names and Types
    print(f"Column Names: {my_data.column_names()}")
    print(f"Column Types: {my_data.column_types()}")

    # Get Metadata and tags associated with this Artifact
    print(f"Meta: {my_data.sageworks_meta()}")
    print(f"Tags: {my_data.sageworks_tags()}")
