"""DataSource: A Factory for DataSources (Athena, RDS, etc)"""

# Local imports
from sageworks.artifacts.data_sources.athena_source import AthenaSource


class DataSource:
    """DataSource: SageWorks DataSource is the best source for your data

    Common Usage:
        my_data = DataSource(data_uuid)
        my_data.summary()
        my_data.details()
        df = my_data.query(f"select * from {data_uuid} limit 5")

    Methods: (implemented by subclasses)
        num_rows(): Return the number of rows for this DataSource
        num_columns(): Return the number of columns
        column_names(): Return the column names
        column_types(): Return the column types
        column_details(): Return the column details
        query(query: str): Returns a pd.DataFrame with the query results
        sample(): Returns a SAMPLED pd.DataFrame from this DataSource
        summary(): Returns a summary of this DataSource
        details(): Returns additional details about this DataSource
        descriptive_stats(): Returns the descriptive stats for each numeric column in this DataSource
        value_counts(): Returns the value counts for each string column in this DataSource
        sageworks_meta(): Returns the SageWorks Metadata for this DataSource
        sageworks_tags(): Returns the SageWorks Tags for this DataSource
        aws_meta(): Returns ALL AWS Metadata for this DataSource
    """

    def __new__(cls, uuid, data_source_type: str = "athena", force_refresh: bool = False):
        """DataSource: A Factory for DataSources (Athena, RDS, etc)
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
            ds = AthenaSource(uuid, force_refresh=force_refresh)
            if ds.exists():
                return ds
            else:
                return AthenaSource(uuid, "sagemaker_featurestore", force_refresh=force_refresh)
        else:
            raise NotImplementedError(f"DataSource type {data_source_type} not implemented")


if __name__ == "__main__":
    """Exercise the DataSource Factory Class"""
    from pprint import pprint

    # Retrieve a DataSource
    my_data = DataSource("abalone_data")

    # Verify that the Athena DataSource exists
    assert my_data.exists()

    # What's my SageWorks UUID
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
    print(f"Column Names: {my_data.column_names()}")
    print(f"Column Types: {my_data.column_types()}")

    # Get Tags associated with this Artifact
    print(f"Tags: {my_data.sageworks_tags()}")

    # Get ALL the AWS Metadata associated with this Artifact
    print("\n\nALL Meta")
    pprint(my_data.aws_meta())

    # Get a SAMPLE of the data
    print(f"Sample Data: {my_data.sample()}")
