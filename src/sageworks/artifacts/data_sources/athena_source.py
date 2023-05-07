"""AthenaSource: SageWorks Data Source accessible through Athena"""
import pandas as pd
import awswrangler as wr
from datetime import datetime
import json

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

        # Refresh our internal catalog metadata
        self.catalog_table_meta = None
        self.refresh()

        # All done
        print(f"AthenaSource Initialized: {self.data_catalog_db}.{self.table_name}")

    def refresh(self, force_fresh: bool = False):
        """Refresh our internal catalog metadata
        Args:
            force_fresh (bool): Force a refresh of the metadata
        """
        _catalog_meta = self.aws_broker.get_metadata(ServiceCategory.DATA_CATALOG, force_fresh=force_fresh)
        self.catalog_table_meta = _catalog_meta[self.data_catalog_db].get(self.table_name)

    def check(self) -> bool:
        """Validation Checks for this Data Source"""

        # We're we able to pull AWS Metadata for this table_name?"""
        if self.catalog_table_meta is None:
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
        params = self.meta().get("Parameters", {})
        return {key: value for key, value in params.items() if "sageworks" in key}

    def upsert_sageworks_meta(self, new_meta: dict):
        """Add SageWorks specific metadata to this Artifact
        Args:
            new_meta (dict): Dictionary of new metadata to add
        """
        # Grab our existing metadata
        meta = self.sageworks_meta()

        # Make sure the new data has keys that are valid
        for key in new_meta.keys():
            if not key.startswith("sageworks_"):
                new_meta[f"sageworks_{key}"] = new_meta.pop(key)

        # Now convert any non-string values to JSON strings
        for key, value in new_meta.items():
            if not isinstance(value, str):
                new_meta[key] = json.dumps(value)

        # Update our existing metadata with the new metadata
        meta.update(new_meta)

        # Store our updated metadata
        wr.catalog.upsert_table_parameters(
            parameters=meta,
            database=self.data_catalog_db,
            table=self.table_name,
            boto3_session=self.boto_session,
        )

    def size(self) -> float:
        """Return the size of this data in MegaBytes"""
        size_in_bytes = sum(wr.s3.size_objects(self.s3_storage_location(), boto3_session=self.boto_session).values())
        size_in_mb = size_in_bytes / 1_000_000
        return size_in_mb

    def meta(self) -> dict:
        """Get the FULL AWS metadata for this artifact"""
        return self.catalog_table_meta

    def aws_url(self):
        """The AWS URL for looking at/querying this data source"""
        return f"https://{self.aws_region}.console.aws.amazon.com/athena/home"

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self.catalog_table_meta["CreateTime"]

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        return self.catalog_table_meta["UpdateTime"]

    def num_rows(self) -> int:
        """Return the number of rows for this Data Source"""
        count_df = self.query(f'select count(*) AS count from "{self.data_catalog_db}"."{self.table_name}"')
        return count_df["count"][0]

    def num_columns(self) -> int:
        """Return the number of columns for this Data Source"""
        return len(self.column_names())

    def column_names(self) -> list[str]:
        """Return the column names for this Athena Table"""
        return [item["Name"] for item in self.catalog_table_meta["StorageDescriptor"]["Columns"]]

    def column_types(self) -> list[str]:
        """Return the column types of the internal AthenaSource"""
        return [item["Type"] for item in self.catalog_table_meta["StorageDescriptor"]["Columns"]]

    def query(self, query: str) -> pd.DataFrame:
        """Query the AthenaSource"""
        df = wr.athena.read_sql_query(sql=query, database=self.data_catalog_db, boto3_session=self.boto_session)
        scanned_bytes = df.query_metadata["Statistics"]["DataScannedInBytes"]
        self.log.info(f"Athena Query successful (scanned bytes: {scanned_bytes})")
        return df

    def s3_storage_location(self) -> str:
        """Get the S3 Storage Location for this Data Source"""
        return self.catalog_table_meta["StorageDescriptor"]["Location"]

    def athena_test_query(self):
        """Validate that Athena Queries are working"""
        query = f"select count(*) as count from {self.table_name}"
        df = wr.athena.read_sql_query(sql=query, database=self.data_catalog_db, boto3_session=self.boto_session)
        scanned_bytes = df.query_metadata["Statistics"]["DataScannedInBytes"]
        self.log.info(f"Athena TEST Query successful (scanned bytes: {scanned_bytes})")

    def sample_df(self, recompute: bool = False) -> pd.DataFrame:
        """Pull a sample of rows from the DataSource
        Args:
            recompute(bool): Recompute the sample (default: False)
        Returns:
            pd.DataFrame: A sample DataFrame from this DataSource
        """

        # First check if we have already computed the quartiles
        if self.sageworks_meta().get("sageworks_sample_rows") and not recompute:
            return pd.read_json(self.sageworks_meta()["sageworks_sample_rows"], orient="records", lines=True)

        # Note: Hardcoded to 100 rows so that metadata storage is consistent
        sample_rows = 100
        num_rows = self.num_rows()
        if num_rows > sample_rows:
            # Bernoulli Sampling has reasonable variance so we're going to +1 the
            # sample percentage and then simply clamp it to 100 rows
            percentage = round(sample_rows * 100.0 / num_rows) + 1
            self.log.warning(f"DataSource has {num_rows} rows.. sampling down to {sample_rows}...")
            query = f"SELECT * FROM {self.table_name} TABLESAMPLE BERNOULLI({percentage})"
        else:
            query = f"SELECT * FROM {self.table_name}"
        sample_df = self.query(query).head(sample_rows)

        # Store the sample_df in our SageWorks metadata
        rows_json = sample_df.to_json(orient="records", lines=True)
        self.upsert_sageworks_meta({"sageworks_sample_rows": rows_json})

        # Return the sample_df
        return sample_df

    def quartiles(self, recompute=False) -> dict[dict]:
        """Compute Quartiles for all the numeric columns in a DataSource
        Args:
            recompute: Recompute the quartiles (default: False)
        Returns:
            dict(dict): A dictionary of quartiles for each column in the form
                 {'col1': {'min': 0, 'q1': 1, 'median': 2, 'q3': 3, 'max': 4},
                  'col2': ...}
        """

        # First check if we have already computed the quartiles
        if self.sageworks_meta().get("sageworks_quartiles") and not recompute:
            return json.loads(self.sageworks_meta()["sageworks_quartiles"])

        # For every column in the table that is numeric, get the quartiles
        self.log.info("Computing Quartiles for all numeric columns (this may take a while)...")
        quartile_data = []
        for column, data_type in zip(self.column_names(), self.column_types()):
            print(column, data_type)
            if data_type in ["bigint", "double", "int", "smallint", "tinyint"]:
                query = (
                    f"SELECT MIN({column}) AS min, "
                    f"approx_percentile({column}, 0.25) AS q1, "
                    f"approx_percentile({column}, 0.5) AS median, "
                    f"approx_percentile({column}, 0.75) AS q3, "
                    f"MAX({column}) AS max FROM {self.table_name}"
                )
                result_df = self.query(query)
                result_df["column_name"] = column
                quartile_data.append(result_df)
        quartile_data = pd.concat(quartile_data).set_index("column_name").to_dict(orient="index")

        # Push the quartile data into our DataSource Metadata
        self.upsert_sageworks_meta({"sageworks_quartiles": quartile_data})

        # Return the quartile data
        return quartile_data

    def details(self) -> dict:
        """Additional Details about this AthenaSource Artifact"""
        details = super().details()
        details["s3_storage_location"] = self.s3_storage_location()
        details.update(self.meta())
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
    from pprint import pprint

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
    print(f"Meta: {my_data.meta()}")
    print(f"Tags: {my_data.sageworks_tags()}")

    # Get a sample of the data
    df = my_data.sample_df()
    print(f"Sample Data: {df.shape}")
    print(df)

    # Get the SageWorks Metadata for this Data Source
    meta = my_data.sageworks_meta()
    print("SageWorks Meta")
    pprint(meta)

    # Add some SageWorks Metadata to this Data Source
    my_data.upsert_sageworks_meta({"sageworks_tags": "abalone:public"})
    print("SageWorks Meta NEW")
    pprint(my_data.sageworks_meta())

    # Get quartiles for all the columns
    quartile_info = my_data.quartiles()
    print("Quartiles")
    pprint(quartile_info)

    # Now delete the AWS artifacts associated with this DataSource
    # print('Deleting SageWorks Data Source...')
    # my_data.delete()
