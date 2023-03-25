"""AthenaSource: SageWorks Data Source accessible through Athena"""
import pandas as pd
import awswrangler as wr
from datetime import datetime

# SageWorks Imports
from sageworks.artifacts.data_sources.data_source import DataSource
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory, AWSServiceBroker


class AthenaSource(DataSource):
    """AthenaSource: SageWorks Data Source accessible through Athena"""
    def __init__(self, table_name, database='sageworks'):
        """AthenaSource Initialization

        Args:
            table_name (str): Name of Athena Table
        """

        # Call superclass init
        super().__init__()

        self.data_catalog_db = database
        self.table_name = table_name

        # Grab an AWS Metadata Broker object and pull information for Data Sources
        self.aws_meta = AWSServiceBroker()
        self.catalog_meta = self.aws_meta.get_metadata(ServiceCategory.DATA_CATALOG)[self.data_catalog_db].get(self.table_name)

        # Grab my tags
        self._tags = self.catalog_meta.get('Parameters', {}).get('tags', []) if self.catalog_meta else []

        # All done
        print(f'AthenaSource Initialized: {self.data_catalog_db}.{self.table_name}')

    def check(self) -> bool:
        """Validation Checks for this Data Source"""

        # We're we able to pull AWS Metadata for this table_name?"""
        if self.catalog_meta is None:
            self.log.critical(f'AthenaSource.check() {self.table_name} not found in SageWorks Metadata!')
            return False
        return True

    def deep_check(self) -> bool:
        """These are more comprehensive checks for this Data Source (may take a LONG TIME)"""

        # Can we run an Athena Test Query
        try:
            self.athena_test_query()
            return True
        except Exception as exc:
            self.log.critical(f"Athena Test Query Failed: {exc}")
            return False

    def uuid(self) -> str:
        """The SageWorks Unique Identifier"""
        return self.table_name

    def _aws_uuid(self) -> str:
        """Internal: An AWS Unique Identifier"""
        return f"CatalogID:{self.catalog_meta['CatalogId']}"

    def size(self) -> bool:
        """Return the size of this data in MegaBytes"""
        size_in_bytes = sum(wr.s3.size_objects(self.s3_storage_location(), boto3_session=self.boto_session).values())
        size_in_mb = round(size_in_bytes / 1_000_000)
        return size_in_mb

    def meta(self):
        """Get the metadata for this artifact"""
        return self.catalog_meta

    def tags(self):
        """Get the tags for this artifact"""
        return self._tags

    def add_tag(self, tag):
        """Add a tag to this artifact"""
        # This ensures no duplicate tags
        self._tags = list(set(self._tags).union([tag]))

    def aws_url(self):
        """The AWS URL for looking at/querying this data source"""
        return 'https://us-west-2.console.aws.amazon.com/athena/home'

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self.catalog_meta['CreateTime']

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        return self.catalog_meta['UpdateTime']

    def num_rows(self) -> int:
        """Return the number of rows for this Data Source"""
        count_df = self.query(f'select count(*) AS count from "{self.data_catalog_db}"."{self.table_name}"')
        return count_df['count'][0]

    def num_columns(self) -> int:
        """Return the number of columns for this Data Source"""
        return len(self.column_names())

    def column_names(self) -> list[str]:
        """Return the column names for this Athena Table"""
        return [item['Name'] for item in self.catalog_meta['StorageDescriptor']['Columns']]

    def query(self, query: str) -> pd.DataFrame:
        """Query the AthenaSource"""
        df = wr.athena.read_sql_query(sql=query, database=self.data_catalog_db, boto3_session=self.boto_session)
        scanned_bytes = df.query_metadata["Statistics"]["DataScannedInBytes"]
        self.log.info(f"Athena Query successful (scanned bytes: {scanned_bytes})")
        return df

    def s3_storage_location(self) -> str:
        """Get the S3 Storage Location for this Data Source"""
        return self.catalog_meta['StorageDescriptor']['Location']

    def athena_test_query(self):
        """Validate that Athena Queries are working"""
        query = f"select count(*) as count from {self.table_name}"
        df = wr.athena.read_sql_query(sql=query, database=self.data_catalog_db, boto3_session=self.boto_session)
        scanned_bytes = df.query_metadata["Statistics"]["DataScannedInBytes"]
        self.log.info(f"Athena TEST Query successful (scanned bytes: {scanned_bytes})")

    def delete(self):
        """Delete the AWS Data Catalog Table and S3 Storage Objects"""

        # Make sure the Feature Group exists
        if not self.check():
            self.log.warning(f"Trying to delete a AthenaSource that doesn't exist: {self.table_name}")

        # FIXME: Add Data Catalog Table and S3 Storage Object Deletion Here
        self.log.info("Add deletion logic")


# Simple test of the AthenaSource functionality
def test():
    """Test for AthenaSource Class"""

    # Retrieve a Data Source
    my_data = AthenaSource('aqsol_data')

    # Verify that the Athena Data Source exists
    assert(my_data.check())

    # What's my SageWorks UUID
    print(f"UUID: {my_data.uuid()}")

    # Get the S3 Storage for this Data Source
    print(f"S3 Storage: {my_data.s3_storage_location()}")

    # What's the size of the data?
    print(f"Size of Data (MB): {my_data.size()}")

    # When was it created and last modified?
    print(f"Created: {my_data.created()}")
    print(f"Modified: {my_data.modified()}")


def more_details():
    """Additional Tests that don't get run automatically"""

    # Retrieve a Data Source
    my_data = AthenaSource('aqsol_data')

    # How many rows and columns?
    num_rows = my_data.num_rows()
    num_columns = my_data.num_columns()
    print(f'Rows: {num_rows} Columns: {num_columns}')

    # What are the column names?
    columns = my_data.column_names()
    print(columns)

    # Get all the metadata associated with this data source
    print(f"Meta: {my_data.meta()}")

    # Get the tags associated with this data source
    print(f"Tags: {my_data.tags()}")

    # Run a query to only pull back a few columns and rows
    column_query = ', '.join(columns[:3])
    query = f'select {column_query} from "{my_data.data_catalog_db}"."{my_data.table_name}" limit 10'
    df = my_data.query(query)
    print(df)

    # Now delete the AWS artifacts associated with this DataSource
    my_data.delete()


if __name__ == "__main__":
    test()
    more_details()
