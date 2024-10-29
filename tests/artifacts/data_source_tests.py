"""Tests for the DataSource/AthenaSource functionality"""

# SageWorks Imports
from sageworks.core.artifacts.athena_source import AthenaSource


def test():
    """Tests for the DataSource/AthenaSource functionality"""
    from pprint import pprint

    # Retrieve our test Data Source
    my_data = AthenaSource("test_data")

    # Verify that the Athena Data Source exists
    assert my_data.exists()

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

    # Get Tags associated with this Artifact
    print(f"Tags: {my_data.get_tags()}")

    # Get ALL the AWS Metadata associated with this Artifact
    print("\n\nALL Meta")
    pprint(my_data.aws_meta())

    # Try to get a data source that doesn't exist
    my_data = AthenaSource("not_exist_data")
    assert not my_data.exists()

    # Try to get a data source that has Mixed Case
    AthenaSource("tEsT_dAtA")  # This will give us some warnings

    # Now delete the AWS artifacts associated with this DataSource
    # print('Deleting SageWorks Data Source...')
    # AthenaSource.managed_delete("tEsT_dAtA")


if __name__ == "__main__":
    test()
