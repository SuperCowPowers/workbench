"""AthenaSource: SageWorks Data Source accessible through Athena"""
import pandas as pd
import awswrangler as wr
from datetime import datetime

# SageWorks Imports
from sageworks.artifacts.data_sources.data_source_abstract import DataSourceAbstract
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory


class AthenaSource(DataSourceAbstract):
    """AthenaSource: SageWorks Data Source accessible through Athena

    Common Usage:
        my_data = AthenaSource(data_uuid, database="sageworks")
        my_data.summary()
        my_data.details()
        df = my_data.query(f"select * from {data_uuid} limit 5")
    """

    def __init__(self, data_uuid, database="sageworks"):
        """AthenaSource Initialization

        Args:
            data_uuid (str): Name of Athena Table
            database (str): Athena Database Name
        """

        # Call superclass init
        super().__init__(data_uuid)

        self.data_catalog_db = database
        self.table_name = data_uuid

        # Grab an AWS Metadata Broker object and pull information for Data Sources
        self.catalog_meta = self.aws_broker.get_metadata(ServiceCategory.DATA_CATALOG)[self.data_catalog_db].get(
            self.table_name
        )

        # All done
        print(f"AthenaSource Initialized: {self.data_catalog_db}.{self.table_name}")

    def check(self) -> bool:
        """Validation Checks for this Data Source"""

        # We're we able to pull AWS Metadata for this table_name?"""
        if self.catalog_meta is None:
            self.log.info(f"AthenaSource.check() {self.table_name} not found in SageWorks Metadata...")
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

    def arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for this artifact"""
        # Grab our SageWorks Role Manager, get our AWS account id, and region for ARN creation
        account_id = self.aws_account_clamp.account_id()
        region = self.aws_account_clamp.region()
        arn = f"arn:aws:glue:{region}:{account_id}:table/{self.data_catalog_db}/{self.table_name}"
        return arn

    def sageworks_meta(self):
        """Get the SageWorks specific metadata for this Artifact"""
        params = self.catalog_meta.get("Parameters", {})
        return {key: value for key, value in params.items() if "sageworks" in key}

    def size(self) -> float:
        """Return the size of this data in MegaBytes"""
        size_in_bytes = sum(wr.s3.size_objects(self.s3_storage_location(), boto3_session=self.boto_session).values())
        size_in_mb = size_in_bytes / 1_000_000
        return size_in_mb

    def aws_meta(self):
        """Get the full AWS metadata for this artifact"""
        return self.catalog_meta

    def aws_url(self):
        """The AWS URL for looking at/querying this data source"""
        return f"https://{self.aws_region}.console.aws.amazon.com/athena/home"

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self.catalog_meta["CreateTime"]

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        return self.catalog_meta["UpdateTime"]

    def num_rows(self) -> int:
        """Return the number of rows for this Data Source"""
        count_df = self.query(f'select count(*) AS count from "{self.data_catalog_db}"."{self.table_name}"')
        return count_df["count"][0]

    def num_columns(self) -> int:
        """Return the number of columns for this Data Source"""
        return len(self.column_names())

    def column_names(self) -> list[str]:
        """Return the column names for this Athena Table"""
        return [item["Name"] for item in self.catalog_meta["StorageDescriptor"]["Columns"]]

    def column_types(self) -> list[str]:
        """Return the column types of the internal AthenaSource"""
        return [item["Type"] for item in self.catalog_meta["StorageDescriptor"]["Columns"]]

    def query(self, query: str) -> pd.DataFrame:
        """Query the AthenaSource"""
        df = wr.athena.read_sql_query(sql=query, database=self.data_catalog_db, boto3_session=self.boto_session)
        scanned_bytes = df.query_metadata["Statistics"]["DataScannedInBytes"]
        self.log.info(f"Athena Query successful (scanned bytes: {scanned_bytes})")
        return df

    def s3_storage_location(self) -> str:
        """Get the S3 Storage Location for this Data Source"""
        return self.catalog_meta["StorageDescriptor"]["Location"]

    def athena_test_query(self):
        """Validate that Athena Queries are working"""
        query = f"select count(*) as count from {self.table_name}"
        df = wr.athena.read_sql_query(sql=query, database=self.data_catalog_db, boto3_session=self.boto_session)
        scanned_bytes = df.query_metadata["Statistics"]["DataScannedInBytes"]
        self.log.info(f"Athena TEST Query successful (scanned bytes: {scanned_bytes})")

    def sample_df(self, max_rows: int = 1000) -> pd.DataFrame:
        """Pull a sample of rows from the DataSource
        Args:
            max_rows (int): Maximum number of rows to pull
        """
        num_rows = self.num_rows()
        if num_rows > max_rows:
            # With Bernoulli Sampling it can give you 0 rows on small samples
            # so we need to make sure we have at least 100 rows for the sample
            # and then we can limit the output to max_rows
            sample_rows = max(max_rows, 100)
            percentage = min(100, round(sample_rows * 100.0 / num_rows))
            self.log.warning(f"DataSource has {num_rows} rows.. sampling down to {max_rows}...")
            query = f"SELECT * FROM {self.table_name} TABLESAMPLE BERNOULLI({percentage})"
        else:
            query = f"SELECT * FROM {self.table_name}"
        return self.query(query).head(max_rows)

    def details(self) -> dict:
        """Additional Details about this AthenaSource Artifact"""
        details = super().details()
        details["s3_storage_location"] = self.s3_storage_location()
        return details

    def delete(self):
        """Delete the AWS Data Catalog Table and S3 Storage Objects"""

        # Make sure the Feature Group exists
        if not self.check():
            self.log.warning(f"Trying to delete a AthenaSource that doesn't exist: {self.table_name}")

        # Delete Data Catalog Table
        self.log.info(f"Deleting DataCatalog Table: {self.data_catalog_db}.{self.table_name}...")
        wr.catalog.delete_table_if_exists(self.data_catalog_db, self.table_name, boto3_session=self.boto_session)

        # Delete S3 Storage Objects
        self.log.info(f"Deleting S3 Storage Object: {self.s3_storage_location()}...")
        wr.s3.delete_objects(self.s3_storage_location(), boto3_session=self.boto_session)


if __name__ == "__main__":
    """Exercise the AthenaSource Class"""

    # Retrieve a Data Source
    my_data = AthenaSource("abalone_data")

    # Verify that the Athena Data Source exists
    assert my_data.check()

    # What's my SageWorks UUID
    print(f"UUID: {my_data.uuid}")

    # What's my AWS ARN and URL
    print(f"AWS ARN: {my_data.arn()}")
    print(f"AWS URL: {my_data.aws_url()}")

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

    # Get a sample of the data
    df = my_data.sample_df(10)
    print(f"Sample Data: {df.shape}")
    print(df)

    # Now delete the AWS artifacts associated with this DataSource
    # print('Deleting SageWorks Data Source...')
    # my_data.delete()
