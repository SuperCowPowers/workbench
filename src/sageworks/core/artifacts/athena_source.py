"""AthenaSource: SageWorks Data Source accessible through Athena"""

from typing import Union
from io import StringIO
import pandas as pd
import awswrangler as wr
from datetime import datetime
import json
import time
import botocore
from pprint import pprint

# SageWorks Imports
from sageworks.core.artifacts.data_source_abstract import DataSourceAbstract
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory
from sageworks.utils.datetime_utils import convert_all_to_iso8601
from sageworks.algorithms.sql import (
    sample_rows,
    value_counts,
    descriptive_stats,
    outliers,
    column_stats,
    correlations,
)
from sageworks.utils.redis_cache import CustomEncoder
from sageworks.utils.aws_utils import sageworks_meta_from_catalog_table_meta


class AthenaSource(DataSourceAbstract):
    """AthenaSource: SageWorks Data Source accessible through Athena

    Common Usage:
        ```
        my_data = AthenaSource(data_uuid, database="sageworks")
        my_data.summary()
        my_data.details()
        df = my_data.query(f"select * from {data_uuid} limit 5")
        ```
    """

    def __init__(self, data_uuid, database="sageworks", force_refresh: bool = False):
        """AthenaSource Initialization

        Args:
            data_uuid (str): Name of Athena Table
            database (str): Athena Database Name (default: sageworks)
            force_refresh (bool): Force refresh of AWS Metadata (default: False)
        """
        # Ensure the data_uuid is a valid name/id
        self.ensure_valid_name(data_uuid)

        # Call superclass init
        super().__init__(data_uuid, database)

        # Flag for metadata cache refresh logic
        self.metadata_refresh_needed = False

        # Setup our AWS Metadata Broker
        self.catalog_table_meta = self.meta_broker.data_source_details(
            data_uuid, self.get_database(), refresh=force_refresh
        )
        if self.catalog_table_meta is None:
            self.log.error(f"Unable to find {self.get_database()}:{self.table} in Glue Catalogs...")

        # Call superclass post init
        super().__post_init__()

        # All done
        self.log.debug(f"AthenaSource Initialized: {self.get_database()}.{self.table}")

    def refresh_meta(self):
        """Refresh our internal AWS Broker catalog metadata"""
        _catalog_meta = self.aws_broker.get_metadata(ServiceCategory.DATA_CATALOG, force_refresh=True)
        self.catalog_table_meta = _catalog_meta[self.get_database()].get(self.table)
        self.metadata_refresh_needed = False

    def exists(self) -> bool:
        """Validation Checks for this Data Source"""

        # Are we able to pull AWS Metadata for this table_name?"""
        # Do we have a valid catalog_table_meta?
        if getattr(self, "catalog_table_meta", None) is None:
            self.log.debug(f"AthenaSource {self.table} not found in SageWorks Metadata...")
            return False
        return True

    def arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for this artifact"""
        # Grab our SageWorks Role Manager, get our AWS account id, and region for ARN creation
        account_id = self.aws_account_clamp.account_id
        region = self.aws_account_clamp.region
        arn = f"arn:aws:glue:{region}:{account_id}:table/{self.get_database()}/{self.table}"
        return arn

    def sageworks_meta(self) -> dict:
        """Get the SageWorks specific metadata for this Artifact"""

        # Sanity Check if we have invalid AWS Metadata
        self.log.info(f"Retrieving SageWorks Metadata for Artifact: {self.uuid}...")
        if self.catalog_table_meta is None:
            if not self.exists():
                self.log.error(f"DataSource {self.uuid} doesn't appear to exist...")
            else:
                self.log.critical(f"Unable to get AWS Metadata for {self.table}")
                self.log.critical("Malformed Artifact! Delete this Artifact and recreate it!")
            return {}

        # Check if we need to refresh our metadata
        if self.metadata_refresh_needed:
            self.refresh_meta()

        # Get the SageWorks Metadata from the Catalog Table Metadata
        return sageworks_meta_from_catalog_table_meta(self.catalog_table_meta)

    def upsert_sageworks_meta(self, new_meta: dict):
        """Add SageWorks specific metadata to this Artifact

        Args:
            new_meta (dict): Dictionary of new metadata to add
        """

        # Give a warning message for keys that don't start with sageworks_
        for key in new_meta.keys():
            if not key.startswith("sageworks_"):
                self.log.warning("Append 'sageworks_' to key names to avoid overwriting AWS meta data")

        # Now convert any non-string values to JSON strings
        for key, value in new_meta.items():
            if not isinstance(value, str):
                new_meta[key] = json.dumps(value, cls=CustomEncoder)

        # Store our updated metadata
        try:
            wr.catalog.upsert_table_parameters(
                parameters=new_meta,
                database=self.get_database(),
                table=self.table,
                boto3_session=self.boto3_session,
            )
            self.metadata_refresh_needed = True
        except botocore.exceptions.ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "InvalidInputException":
                self.log.error(f"Unable to upsert metadata for {self.table}")
                self.log.error("Probably because the metadata is too large")
                self.log.error(new_meta)
            elif error_code == "ConcurrentModificationException":
                self.log.warning("ConcurrentModificationException... trying again...")
                time.sleep(5)
                wr.catalog.upsert_table_parameters(
                    parameters=new_meta,
                    database=self.get_database(),
                    table=self.table,
                    boto3_session=self.boto3_session,
                )
            else:
                self.log.critical(f"Failed to upsert metadata: {e}")
                self.log.critical(f"{self.uuid} is Malformed! Delete this Artifact and recreate it!")
        except Exception as e:
            self.log.critical(f"Failed to upsert metadata: {e}")
            self.log.critical(f"{self.uuid} is Malformed! Delete this Artifact and recreate it!")

    def size(self) -> float:
        """Return the size of this data in MegaBytes"""
        size_in_bytes = sum(wr.s3.size_objects(self.s3_storage_location(), boto3_session=self.boto3_session).values())
        size_in_mb = size_in_bytes / 1_000_000
        return size_in_mb

    def aws_meta(self) -> dict:
        """Get the FULL AWS metadata for this artifact"""
        return self.catalog_table_meta

    def aws_url(self):
        """The AWS URL for looking at/querying this data source"""
        sageworks_details = self.sageworks_meta().get("sageworks_details", {})
        return sageworks_details.get("aws_url", "unknown")

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self.catalog_table_meta["CreateTime"]

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        return self.catalog_table_meta["UpdateTime"]

    def num_rows(self) -> int:
        """Return the number of rows for this Data Source"""
        count_df = self.query(f'select count(*) AS sageworks_count from "{self.get_database()}"."{self.table}"')
        return count_df["sageworks_count"][0] if count_df is not None else 0

    def num_columns(self) -> int:
        """Return the number of columns for this Data Source"""
        return len(self.columns)

    @property
    def columns(self) -> list[str]:
        """Return the column names for this Athena Table"""
        return [item["Name"] for item in self.catalog_table_meta["StorageDescriptor"]["Columns"]]

    @property
    def column_types(self) -> list[str]:
        """Return the column types of the internal AthenaSource"""
        return [item["Type"] for item in self.catalog_table_meta["StorageDescriptor"]["Columns"]]

    def query(self, query: str) -> Union[pd.DataFrame, None]:
        """Query the AthenaSource

        Args:
            query (str): The query to run against the AthenaSource

        Returns:
            pd.DataFrame: The results of the query
        """
        self.log.debug(f"Executing Query: {query}...")
        try:
            df = wr.athena.read_sql_query(
                sql=query,
                database=self.get_database(),
                ctas_approach=False,
                boto3_session=self.boto3_session,
            )
            scanned_bytes = df.query_metadata["Statistics"]["DataScannedInBytes"]
            if scanned_bytes > 0:
                self.log.debug(f"Athena Query successful (scanned bytes: {scanned_bytes})")
            return df
        except wr.exceptions.QueryFailed as e:
            self.log.critical(f"Failed to execute query: {e}")
            return None

    def execute_statement(self, query: str, silence_errors: bool = False):
        """Execute a non-returning SQL statement in Athena

        Args:
            query (str): The query to run against the AthenaSource
            silence_errors (bool): Silence errors (default: False)
        """
        try:
            # Start the query execution
            query_execution_id = wr.athena.start_query_execution(
                sql=query,
                database=self.get_database(),
                boto3_session=self.boto3_session,
            )
            self.log.debug(f"QueryExecutionId: {query_execution_id}")

            # Wait for the query to complete
            wr.athena.wait_query(query_execution_id=query_execution_id, boto3_session=self.boto3_session)
            self.log.debug(f"Statement executed successfully: {query_execution_id}")
        except wr.exceptions.QueryFailed as e:
            if "AlreadyExistsException" in str(e):
                self.log.warning(f"Table already exists. Ignoring: {e}")
            else:
                if not silence_errors:
                    self.log.error(f"Failed to execute statement: {e}")
                raise
        except botocore.exceptions.ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "InvalidRequestException":
                self.log.error(f"Invalid Query: {query}")
            else:
                self.log.error(f"Failed to execute statement: {e}")
            raise

    def s3_storage_location(self) -> str:
        """Get the S3 Storage Location for this Data Source"""
        return self.catalog_table_meta["StorageDescriptor"]["Location"]

    def athena_test_query(self):
        """Validate that Athena Queries are working"""
        query = f"select count(*) as sageworks_count from {self.table}"
        df = wr.athena.read_sql_query(
            sql=query,
            database=self.get_database(),
            ctas_approach=False,
            boto3_session=self.boto3_session,
        )
        scanned_bytes = df.query_metadata["Statistics"]["DataScannedInBytes"]
        self.log.info(f"Athena TEST Query successful (scanned bytes: {scanned_bytes})")

    def sample_impl(self) -> pd.DataFrame:
        """Pull a sample of rows from the DataSource

        Returns:
            pd.DataFrame: A sample DataFrame for an Athena DataSource
        """

        # Call the SQL function to pull a sample of the rows
        return sample_rows.sample_rows(self)

    def descriptive_stats(self, recompute: bool = False) -> dict[dict]:
        """Compute Descriptive Stats for all the numeric columns in a DataSource

        Args:
            recompute (bool): Recompute the descriptive stats (default: False)

        Returns:
            dict(dict): A dictionary of descriptive stats for each column in the form
                 {'col1': {'min': 0, 'q1': 1, 'median': 2, 'q3': 3, 'max': 4},
                  'col2': ...}
        """

        # First check if we have already computed the descriptive stats
        stat_dict = self.sageworks_meta().get("sageworks_descriptive_stats")
        if stat_dict and not recompute:
            return stat_dict

        # Call the SQL function to compute descriptive stats
        stat_dict = descriptive_stats.descriptive_stats(self)

        # Push the descriptive stat data into our DataSource Metadata
        self.upsert_sageworks_meta({"sageworks_descriptive_stats": stat_dict})

        # Return the descriptive stats
        return stat_dict

    def outliers_impl(self, scale: float = 1.5, use_stddev=False) -> pd.DataFrame:
        """Compute outliers for all the numeric columns in a DataSource

        Args:
            scale (float): The scale to use for the IQR (default: 1.5)
            use_stddev (bool): Use Standard Deviation instead of IQR (default: False)

        Returns:
            pd.DataFrame: A DataFrame of outliers from this DataSource

        Notes:
            Uses the IQR * 1.5 (~= 2.5 Sigma) (use 1.7 for ~= 3 Sigma)
            The scale parameter can be adjusted to change the IQR multiplier
        """

        # Compute outliers using the SQL Outliers class
        sql_outliers = outliers.Outliers()
        return sql_outliers.compute_outliers(self, scale=scale, use_stddev=use_stddev)

    def smart_sample(self, recompute: bool = False) -> pd.DataFrame:
        """Get a smart sample dataframe for this DataSource

        Args:
            recompute (bool): Recompute the smart sample (default: False)

        Returns:
            pd.DataFrame: A combined DataFrame of sample data + outliers
        """

        # Check if we have cached smart_sample data
        storage_key = f"data_source:{self.uuid}:smart_sample"
        if not recompute and self.data_storage.get(storage_key):
            return pd.read_json(StringIO(self.data_storage.get(storage_key)))

        # Compute/recompute the smart sample
        self.log.important(f"Computing Smart Sample {self.uuid}...")

        # Outliers DataFrame
        outlier_rows = self.outliers(recompute=recompute)

        # Sample DataFrame
        sample_rows = self.sample(recompute=recompute)
        sample_rows["outlier_group"] = "sample"

        # Combine the sample rows with the outlier rows
        all_rows = pd.concat([outlier_rows, sample_rows]).reset_index(drop=True)

        # Drop duplicates
        all_except_outlier_group = [col for col in all_rows.columns if col != "outlier_group"]
        all_rows = all_rows.drop_duplicates(subset=all_except_outlier_group, ignore_index=True)

        # Cache the smart_sample data
        self.data_storage.set(storage_key, all_rows.to_json())

        # Return the smart_sample data
        return all_rows

    def correlations(self, recompute: bool = False) -> dict[dict]:
        """Compute Correlations for all the numeric columns in a DataSource

        Args:
            recompute (bool): Recompute the column stats (default: False)

        Returns:
            dict(dict): A dictionary of correlations for each column in this format
                 {'col1': {'col2': 0.5, 'col3': 0.9, 'col4': 0.4, ...},
                  'col2': {'col1': 0.5, 'col3': 0.8, 'col4': 0.3, ...}}
        """

        # First check if we have already computed the correlations
        correlations_dict = self.sageworks_meta().get("sageworks_correlations")
        if correlations_dict and not recompute:
            return correlations_dict

        # Call the SQL function to compute correlations
        correlations_dict = correlations.correlations(self)

        # Push the correlation data into our DataSource Metadata
        self.upsert_sageworks_meta({"sageworks_correlations": correlations_dict})

        # Return the correlation data
        return correlations_dict

    def column_stats(self, recompute: bool = False) -> dict[dict]:
        """Compute Column Stats for all the columns in a DataSource

        Args:
            recompute (bool): Recompute the column stats (default: False)

        Returns:
            dict(dict): A dictionary of stats for each column this format
            NB: String columns will NOT have num_zeros, descriptive_stats or correlation data
                {'col1': {'dtype': 'string', 'unique': 4321, 'nulls': 12},
                 'col2': {'dtype': 'int', 'unique': 4321, 'nulls': 12, 'num_zeros': 100,
                          'descriptive_stats': {...}, 'correlations': {...}},
                 ...}
        """

        # First check if we have already computed the column stats
        columns_stats_dict = self.sageworks_meta().get("sageworks_column_stats")
        if columns_stats_dict and not recompute:
            return columns_stats_dict

        # Call the SQL function to compute column stats
        column_stats_dict = column_stats.column_stats(self, recompute=recompute)

        # Push the column stats data into our DataSource Metadata
        self.upsert_sageworks_meta({"sageworks_column_stats": column_stats_dict})

        # Return the column stats data
        return column_stats_dict

    def value_counts(self, recompute: bool = False) -> dict[dict]:
        """Compute 'value_counts' for all the string columns in a DataSource

        Args:
            recompute (bool): Recompute the value counts (default: False)

        Returns:
            dict(dict): A dictionary of value counts for each column in the form
                 {'col1': {'value_1': 42, 'value_2': 16, 'value_3': 9,...},
                  'col2': ...}
        """

        # First check if we have already computed the value counts
        value_counts_dict = self.sageworks_meta().get("sageworks_value_counts")
        if value_counts_dict and not recompute:
            return value_counts_dict

        # Call the SQL function to compute value_counts
        value_count_dict = value_counts.value_counts(self)

        # Push the value_count data into our DataSource Metadata
        self.upsert_sageworks_meta({"sageworks_value_counts": value_count_dict})

        # Return the value_count data
        return value_count_dict

    def details(self, recompute: bool = False) -> dict[dict]:
        """Additional Details about this AthenaSource Artifact

        Args:
            recompute (bool): Recompute the details (default: False)

        Returns:
            dict(dict): A dictionary of details about this AthenaSource
        """

        # Check if we have cached version of the DataSource Details
        storage_key = f"data_source:{self.uuid}:details"
        cached_details = self.data_storage.get(storage_key)
        if cached_details and not recompute:
            return cached_details

        self.log.info(f"Recomputing DataSource Details ({self.uuid})...")

        # Get the details from the base class
        details = super().details()

        # Compute additional details
        details["s3_storage_location"] = self.s3_storage_location()
        details["storage_type"] = "athena"

        # Compute our AWS URL
        query = f"select * from {self.get_database()}.{self.table} limit 10"
        query_exec_id = wr.athena.start_query_execution(
            sql=query, database=self.get_database(), boto3_session=self.boto3_session
        )
        base_url = "https://console.aws.amazon.com/athena/home"
        details["aws_url"] = f"{base_url}?region={self.aws_region}#query/history/{query_exec_id}"

        # Push the aws_url data into our DataSource Metadata
        self.upsert_sageworks_meta({"sageworks_details": {"aws_url": details["aws_url"]}})

        # Convert any datetime fields to ISO-8601 strings
        details = convert_all_to_iso8601(details)

        # Add the column stats
        details["column_stats"] = self.column_stats()

        # Cache the details
        self.data_storage.set(storage_key, details)

        # Return the details data
        return details

    def delete(self):
        """Delete the AWS Data Catalog Table and S3 Storage Objects"""

        # Make sure the AthenaSource exists
        if not self.exists():
            self.log.warning(f"Trying to delete a AthenaSource that doesn't exist: {self.table}")

        # Delete any views associated with this AthenaSource
        self.delete_views()

        # Delete Data Catalog Table
        self.log.info(f"Deleting DataCatalog Table: {self.get_database()}.{self.table}...")
        wr.catalog.delete_table_if_exists(self.get_database(), self.table, boto3_session=self.boto3_session)

        # Delete S3 Storage Objects (if they exist)
        try:
            # Make sure we add the trailing slash
            s3_path = self.s3_storage_location()
            s3_path = s3_path if s3_path.endswith("/") else f"{s3_path}/"

            self.log.info(f"Deleting S3 Storage Objects: {s3_path}...")
            wr.s3.delete_objects(s3_path, boto3_session=self.boto3_session)
        except Exception as e:
            self.log.error(f"Failed to delete S3 Storage Objects: {e}")
            self.log.warning("Malformed Artifact... good thing it's being deleted...")

        # Delete any data in the Cache
        for key in self.data_storage.list_subkeys(f"data_source:{self.uuid}:"):
            self.log.info(f"Deleting Cache Key {key}...")
            self.data_storage.delete(key)

    def delete_views(self):
        """Delete any views associated with this FeatureSet"""
        from sageworks.core.views.view_utils import delete_views_and_supplemental_data

        delete_views_and_supplemental_data(self)


if __name__ == "__main__":
    """Exercise the AthenaSource Class"""

    # Retrieve a Data Source
    my_data = AthenaSource("abalone_data")

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

    # Column Names and Types
    print(f"Column Names: {my_data.columns}")
    print(f"Column Types: {my_data.column_types}")
    print(f"Column Details: {my_data.column_details()}")

    # Get the input for this Artifact
    print(f"Input: {my_data.get_input()}")

    # Get Tags associated with this Artifact
    print(f"Tags: {my_data.get_tags()}")

    # Get a sample of the data
    my_df = my_data.sample()
    print(f"Sample Data: {my_df.shape}")
    print(my_df)

    # Get the SageWorks Metadata for this Data Source
    meta = my_data.sageworks_meta()
    print("\nSageWorks Meta")
    pprint(meta)

    # Get details for this Data Source
    my_details = my_data.details()
    print("\nDetails")
    pprint(my_details)

    # Get outliers for numeric columns
    my_outlier_df = my_data.outliers()
    print("\nOutliers")
    print(my_outlier_df)

    # Get a smart sample for numeric columns
    smart_sample = my_data.smart_sample()
    print("\nSmart Sample")
    print(smart_sample)

    # Get descriptive stats for numeric columns
    stat_info = my_data.descriptive_stats()
    print("\nDescriptive Stats")
    pprint(stat_info)

    # Get value_counts for string columns
    value_count_info = my_data.value_counts()
    print("\nValue Counts")
    pprint(value_count_info)

    # Get correlations for numeric columns
    my_correlation_df = my_data.correlations()
    print("\nCorrelations")
    print(my_correlation_df)

    # Get ALL the AWS Metadata associated with this Artifact
    print("\n\nALL Meta")
    pprint(my_data.aws_meta())

    # Get the display columns
    print("\n\nDisplay Columns")
    print(my_data.view("display").columns)

    # Set the computation columns
    print("\n\nSet Computation Columns")
    print(my_data.set_computation_columns(my_data.columns))

    # Get the computation columns
    print("\n\nComputation Columns")
    print(my_data.view("computation").columns)

    # Test a Data Source that doesn't exist
    # The rest of the tests are Disabled for now
    """
    print("\n\nTesting a Data Source that does not exist...")
    my_data = AthenaSource("does_not_exist")
    assert not my_data.exists()
    my_data.sageworks_meta()

    # Now delete the AWS artifacts associated with this DataSource
    print('Deleting SageWorks Data Source...')
    my_data.delete()
    """
