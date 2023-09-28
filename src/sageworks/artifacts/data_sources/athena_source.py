"""AthenaSource: SageWorks Data Source accessible through Athena"""
import pandas as pd
import awswrangler as wr
from datetime import datetime
import json
import botocore
from pprint import pprint

# SageWorks Imports
from sageworks.artifacts.data_sources.data_source_abstract import DataSourceAbstract
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory
from sageworks.utils.iso_8601 import convert_all_to_iso8601
from sageworks.algorithms.sql import (
    sample_rows,
    value_counts,
    descriptive_stats,
    outliers,
    column_stats,
    correlations,
)
from sageworks.utils.pandas_utils import NumpyEncoder


class AthenaSource(DataSourceAbstract):
    """AthenaSource: SageWorks Data Source accessible through Athena

    Common Usage:
        my_data = AthenaSource(data_uuid, database="sageworks")
        my_data.summary()
        my_data.details()
        df = my_data.query(f"select * from {data_uuid} limit 5")
    """

    def __init__(self, data_uuid, database="sageworks", force_refresh: bool = False):
        """AthenaSource Initialization
        Args:
            data_uuid (str): Name of Athena Table
            database (str): Athena Database Name (default: sageworks)
            force_refresh (bool): Force refresh of AWS Metadata (default: False)
        """

        # Call superclass init
        super().__init__(data_uuid)

        self.data_catalog_db = database
        self.table_name = data_uuid

        # Setup our AWS Broker catalog metadata
        _catalog_meta = self.aws_broker.get_metadata(ServiceCategory.DATA_CATALOG, force_refresh=force_refresh)
        self.catalog_table_meta = _catalog_meta[self.data_catalog_db].get(self.uuid)

        # All done
        self.log.debug(f"AthenaSource Initialized: {self.data_catalog_db}.{self.table_name}")

    def refresh_meta(self):
        """Refresh our internal AWS Broker catalog metadata"""
        _catalog_meta = self.aws_broker.get_metadata(ServiceCategory.DATA_CATALOG, force_refresh=True)
        self.catalog_table_meta = _catalog_meta[self.data_catalog_db].get(self.table_name)

    def exists(self) -> bool:
        """Validation Checks for this Data Source"""

        # We're we able to pull AWS Metadata for this table_name?"""
        if self.catalog_table_meta is None:
            self.log.info(f"AthenaSource {self.table_name} not found in SageWorks Metadata...")
            return False
        return True

    def arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for this artifact"""
        # Grab our SageWorks Role Manager, get our AWS account id, and region for ARN creation
        account_id = self.aws_account_clamp.account_id
        region = self.aws_account_clamp.region
        arn = f"arn:aws:glue:{region}:{account_id}:table/{self.data_catalog_db}/{self.table_name}"
        return arn

    def sageworks_meta(self) -> dict:
        """Get the SageWorks specific metadata for this Artifact"""
        # Sanity Check if we have invalid AWS Metadata
        if self.aws_meta() is None:
            self.log.critical(f"Unable to get AWS Metadata for {self.table_name}")
            self.log.critical("Malformed Artifact! Delete this Artifact and recreate it!")
            return {}
        params = self.aws_meta().get("Parameters", {})
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
                new_meta[key] = json.dumps(value, cls=NumpyEncoder)

        # Update our existing metadata with the new metadata
        meta.update(new_meta)

        # Store our updated metadata
        try:
            wr.catalog.upsert_table_parameters(
                parameters=meta,
                database=self.data_catalog_db,
                table=self.table_name,
                boto3_session=self.boto_session,
            )
        except botocore.exceptions.ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "InvalidInputException":
                self.log.error(f"Unable to upsert metadata for {self.table_name}")
                self.log.error("Probably because the metadata is too large")
                self.log.error(new_meta)
            else:
                raise e

    def size(self) -> float:
        """Return the size of this data in MegaBytes"""
        size_in_bytes = sum(wr.s3.size_objects(self.s3_storage_location(), boto3_session=self.boto_session).values())
        size_in_mb = size_in_bytes / 1_000_000
        return size_in_mb

    def aws_meta(self) -> dict:
        """Get the FULL AWS metadata for this artifact"""
        return self.catalog_table_meta

    def aws_url(self):
        """The AWS URL for looking at/querying this data source"""
        return self.details().get("aws_url", "unknown")

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
        df = wr.athena.read_sql_query(
            sql=query,
            database=self.data_catalog_db,
            ctas_approach=False,
            boto3_session=self.boto_session,
        )
        scanned_bytes = df.query_metadata["Statistics"]["DataScannedInBytes"]
        self.log.debug(f"Athena Query successful (scanned bytes: {scanned_bytes})")
        return df

    def s3_storage_location(self) -> str:
        """Get the S3 Storage Location for this Data Source"""
        return self.catalog_table_meta["StorageDescriptor"]["Location"]

    def athena_test_query(self):
        """Validate that Athena Queries are working"""
        query = f"select count(*) as count from {self.table_name}"
        df = wr.athena.read_sql_query(
            sql=query,
            database=self.data_catalog_db,
            ctas_approach=False,
            boto3_session=self.boto_session,
        )
        scanned_bytes = df.query_metadata["Statistics"]["DataScannedInBytes"]
        self.log.debug(f"Athena TEST Query successful (scanned bytes: {scanned_bytes})")

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
            recompute(bool): Recompute the descriptive stats (default: False)
        Returns:
            dict(dict): A dictionary of descriptive stats for each column in the form
                 {'col1': {'min': 0, 'q1': 1, 'median': 2, 'q3': 3, 'max': 4},
                  'col2': ...}
        """

        # First check if we have already computed the descriptive stats
        if self.sageworks_meta().get("sageworks_descriptive_stats") and not recompute:
            return json.loads(self.sageworks_meta()["sageworks_descriptive_stats"])

        # Call the SQL function to compute descriptive stats
        stat_dict = descriptive_stats.descriptive_stats(self)

        # Push the descriptive stat data into our DataSource Metadata
        self.upsert_sageworks_meta({"sageworks_descriptive_stats": stat_dict})

        # Return the descriptive stats
        return stat_dict

    def outliers_impl(self, scale: float = 1.5, use_stddev=False, recompute: bool = False) -> pd.DataFrame:
        """Compute outliers for all the numeric columns in a DataSource
        Args:
            scale(float): The scale to use for the IQR (default: 1.5)
            use_stddev(bool): Use Standard Deviation instead of IQR (default: False)
            recompute(bool): Recompute the outliers (default: False)
        Returns:
            pd.DataFrame: A DataFrame of outliers from this DataSource
        Notes:
            Uses the IQR * 1.5 (~= 2.5 Sigma) (use 1.7 for ~= 3 Sigma)
            The scale parameter can be adjusted to change the IQR multiplier
        """

        # Compute outliers using the SQL Outliers class
        sql_outliers = outliers.Outliers()
        return sql_outliers.compute_outliers(self, scale=scale, use_stddev=use_stddev)

    def smart_sample(self) -> pd.DataFrame:
        """Get a smart sample dataframe for this DataSource
        Note:
            smart = sample data + outliers for the DataSource"""

        # Outliers DataFrame
        outlier_rows = self.outliers()

        # Sample DataFrame
        sample_rows = self.sample()
        sample_rows["outlier_group"] = "sample"

        # Combine the sample rows with the outlier rows
        all_rows = pd.concat([outlier_rows, sample_rows]).reset_index(drop=True)

        # Drop duplicates
        all_except_outlier_group = [col for col in all_rows.columns if col != "outlier_group"]
        all_rows = all_rows.drop_duplicates(subset=all_except_outlier_group, ignore_index=True)
        return all_rows

    def correlations(self, recompute: bool = False) -> dict[dict]:
        """Compute Correlations for all the numeric columns in a DataSource
        Args:
            recompute(bool): Recompute the column stats (default: False)
        Returns:
            dict(dict): A dictionary of correlations for each column in this format
                 {'col1': {'col2': 0.5, 'col3': 0.9, 'col4': 0.4, ...},
                  'col2': {'col1': 0.5, 'col3': 0.8, 'col4': 0.3, ...}}
        """

        # First check if we have already computed the correlations
        if self.sageworks_meta().get("sageworks_correlations") and not recompute:
            return json.loads(self.sageworks_meta()["sageworks_correlations"])

        # Call the SQL function to compute correlations
        correlations_dict = correlations.correlations(self)

        # Push the correlation data into our DataSource Metadata
        self.upsert_sageworks_meta({"sageworks_correlations": correlations_dict})

        # Return the correlation data
        return correlations_dict

    def column_stats(self, recompute: bool = False) -> dict[dict]:
        """Compute Column Stats for all the columns in a DataSource
        Args:
            recompute(bool): Recompute the column stats (default: False)
        Returns:
            dict(dict): A dictionary of stats for each column this format
            NB: String columns will NOT have num_zeros, descriptive_stats or correlation data
             {'col1': {'dtype': 'string', 'unique': 4321, 'nulls': 12},
              'col2': {'dtype': 'int', 'unique': 4321, 'nulls': 12, 'num_zeros': 100,
                       'descriptive_stats': {...}, 'correlations': {...}},
              ...}
        """

        # First check if we have already computed the column stats
        if self.sageworks_meta().get("sageworks_column_stats") and not recompute:
            return json.loads(self.sageworks_meta()["sageworks_column_stats"])

        # Call the SQL function to compute column stats
        column_stats_dict = column_stats.column_stats(self)

        # Push the column stats data into our DataSource Metadata
        self.upsert_sageworks_meta({"sageworks_column_stats": column_stats_dict})

        # Return the column stats data
        return column_stats_dict

    def value_counts(self, recompute: bool = False) -> dict[dict]:
        """Compute 'value_counts' for all the string columns in a DataSource
        Args:
            recompute(bool): Recompute the value counts (default: False)
        Returns:
            dict(dict): A dictionary of value counts for each column in the form
                 {'col1': {'value_1': 42, 'value_2': 16, 'value_3': 9,...},
                  'col2': ...}
        """

        # First check if we have already computed the value counts
        if self.sageworks_meta().get("sageworks_value_counts") and not recompute:
            return json.loads(self.sageworks_meta()["sageworks_value_counts"])

        # Call the SQL function to compute value_counts
        value_count_dict = value_counts.value_counts(self)

        # Push the value_count data into our DataSource Metadata
        self.upsert_sageworks_meta({"sageworks_value_counts": value_count_dict})

        # Return the value_count data
        return value_count_dict

    def details(self, recompute: bool = False) -> dict[dict]:
        """Additional Details about this AthenaSource Artifact
        Args:
            recompute(bool): Recompute the details (default: False)
        Returns:
            dict(dict): A dictionary of details about this AthenaSource
        """

        # First check if we have already computed the details
        if self.sageworks_meta().get("sageworks_details") and not recompute:
            return json.loads(self.sageworks_meta()["sageworks_details"])

        # Get the details from the base class
        details = super().details()

        # Compute additional details
        details["s3_storage_location"] = self.s3_storage_location()
        details["storage_type"] = "athena"

        # Compute our AWS URL
        query = f"select * from {self.data_catalog_db}.{self.table_name} limit 10"
        query_exec_id = wr.athena.start_query_execution(
            sql=query, database=self.data_catalog_db, boto3_session=self.boto_session
        )
        base_url = "https://console.aws.amazon.com/athena/home"
        details["aws_url"] = f"{base_url}?region={self.aws_region}#query/history/{query_exec_id}"

        # Convert any datetime fields to ISO-8601 strings
        details = convert_all_to_iso8601(details)

        # Push the details data into our DataSource Metadata
        self.upsert_sageworks_meta({"sageworks_details": details})

        # Return the details data
        return details

    def delete(self):
        """Delete the AWS Data Catalog Table and S3 Storage Objects"""

        # Make sure the Feature Group exists
        if not self.exists():
            self.log.warning(f"Trying to delete a AthenaSource that doesn't exist: {self.table_name}")

        # Delete Data Catalog Table
        self.log.info(f"Deleting DataCatalog Table: {self.data_catalog_db}.{self.table_name}...")
        wr.catalog.delete_table_if_exists(self.data_catalog_db, self.table_name, boto3_session=self.boto_session)

        # Delete S3 Storage Objects (if they exist)
        try:
            self.log.info(f"Deleting S3 Storage Object: {self.s3_storage_location()}...")
            wr.s3.delete_objects(self.s3_storage_location(), boto3_session=self.boto_session)
        except TypeError:
            self.log.warning("Malformed Artifact... good thing it's being deleted...")


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
    print(f"Column Names: {my_data.column_names()}")
    print(f"Column Types: {my_data.column_types()}")
    print(f"Column Details: {my_data.column_details()}")

    # Get the input for this Artifact
    print(f"Input: {my_data.get_input()}")

    # Get Tags associated with this Artifact
    print(f"Tags: {my_data.sageworks_tags()}")

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

    # Test an Data Source that doesn't exist
    print("\n\nTesting a Data Source that does not exist...")
    my_data = AthenaSource("does_not_exist")
    assert not my_data.exists()
    my_data.sageworks_meta()

    # Now delete the AWS artifacts associated with this DataSource
    # print('Deleting SageWorks Data Source...')
    # my_data.delete()
