"""AthenaSource: SageWorks Data Source accessible through Athena"""
import pandas as pd
import numpy as np
import awswrangler as wr
from datetime import datetime
import json

# SageWorks Imports
from sageworks.artifacts.data_sources.data_source_abstract import DataSourceAbstract
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory
from sageworks.utils.iso_8601 import convert_all_to_iso8601


class AthenaSource(DataSourceAbstract):
    """AthenaSource: SageWorks Data Source accessible through Athena

    Common Usage:
        my_data = AthenaSource(data_uuid, database="sageworks")
        my_data.summary()
        my_data.details()
        df = my_data.query(f"select * from {data_uuid} limit 5")
    """

    def __init__(self, data_uuid, database="sageworks", force_refresh=False):
        """AthenaSource Initialization

        Args:
            data_uuid (str): Name of Athena Table
            database (str): Athena Database Name
            force_refresh (bool): Force a refresh of the metadata
        """

        # Call superclass init
        super().__init__(data_uuid)

        self.data_catalog_db = database
        self.table_name = data_uuid

        # Refresh our internal catalog metadata
        self.catalog_table_meta = self._refresh_broker(force_refresh)

        # All done
        self.log.debug(f"AthenaSource Initialized: {self.data_catalog_db}.{self.table_name}")

    def _refresh_broker(self, force_refresh=False):
        """Internal: Refresh our internal catalog metadata
        Args:
            force_refresh (bool): Force a refresh of the metadata
        """
        _catalog_meta = self.aws_broker.get_metadata(ServiceCategory.DATA_CATALOG, force_refresh=force_refresh)
        return _catalog_meta[self.data_catalog_db].get(self.table_name)

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
        self.log.info(f"Athena Query successful (scanned bytes: {scanned_bytes})")
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
        self.log.info(f"Athena TEST Query successful (scanned bytes: {scanned_bytes})")

    def sample_df(self, recompute: bool = False) -> pd.DataFrame:
        """Pull a sample of rows from the DataSource
        Args:
            recompute(bool): Recompute the sample (default: False)
        Returns:
            pd.DataFrame: A sample DataFrame from this DataSource
        """

        # First check if we have already computed the sample dataframe
        if self.sageworks_meta().get("sageworks_sample_rows") and not recompute:
            return pd.read_json(
                self.sageworks_meta()["sageworks_sample_rows"],
                orient="records",
                lines=True,
            )

        # Note: Hardcoded to 100 rows so that metadata storage is consistent
        sample_rows = 100
        num_rows = self.details()["num_rows"]
        if num_rows > sample_rows:
            # Bernoulli Sampling has reasonable variance, so we're going to +1 the
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

    def quartiles(self, recompute: bool = False) -> dict[dict]:
        """Compute Quartiles for all the numeric columns in a DataSource
        Args:
            recompute(bool): Recompute the quartiles (default: False)
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
                    f'SELECT MIN("{column}") AS min, '
                    f'approx_percentile("{column}", 0.25) AS q1, '
                    f'approx_percentile("{column}", 0.5) AS median, '
                    f'approx_percentile("{column}", 0.75) AS q3, '
                    f'MAX("{column}") AS max FROM {self.table_name}'
                )
                result_df = self.query(query)
                result_df["column_name"] = column
                quartile_data.append(result_df)
        quartile_dict = pd.concat(quartile_data).set_index("column_name").to_dict(orient="index")

        # Push the quartile data into our DataSource Metadata
        self.upsert_sageworks_meta({"sageworks_quartiles": quartile_dict})

        # Return the quartile data
        return quartile_dict

    def _outlier_dfs(self, column: str, lower_bound: float, upper_bound: float) -> pd.DataFrame:
        """Internal method to compute outliers for a numeric column
        Returns:
            pd.DataFrame, pd.DataFrame: A DataFrame for lower outliers and a DataFrame for upper outliers
        """

        # Get lower outlier bound
        query = f"SELECT * from {self.table_name} where {column} < {lower_bound} order by {column} limit 5"
        lower_df = self.query(query)

        # Check for no results
        if lower_df.shape[0] == 0:
            lower_df = None

        # Get upper outlier bound
        query = f"SELECT * from {self.table_name} where {column} > {upper_bound} order by {column} desc limit 5"
        upper_df = self.query(query)

        # Check for no results
        if upper_df.shape[0] == 0:
            upper_df = None

        # Return the lower and upper outlier DataFrames
        return lower_df, upper_df

    def outliers(self, scale: float = 1.7, recompute: bool = False, project: bool = False) -> pd.DataFrame:
        """Compute outliers for all the numeric columns in a DataSource
        Args:
            scale(float): The scale to use for the IQR (default: 1.7)
            recompute(bool): Recompute the outliers (default: False)
            project(bool): Project the outliers onto an x,y plane (default: False)
        Returns:
            pd.DataFrame: A DataFrame of outliers from this DataSource
        Notes:
            Uses the IQR * 1.7 (~= 3 Sigma) method to compute outliers
            The scale parameter can be adjusted to change the IQR multiplier
        """

        # First check if we have already computed the outliers
        if self.sageworks_meta().get("sageworks_outliers") and not recompute:
            return pd.read_json(
                self.sageworks_meta()["sageworks_outliers"],
                orient="records",
                lines=True,
            )

        # Grab the quartiles for this DataSource
        quartiles = self.quartiles()

        # For every column in the table that is numeric get the outliers
        self.log.info("Computing outliers for all columns (this may take a while)...")
        cluster = 0
        outlier_df_list = []
        outlier_features = []
        num_rows = self.details()["num_rows"]
        outlier_count = num_rows * 0.001  # 0.1% of the total rows
        value_count_info = self.value_counts()
        for column, data_type in zip(self.column_names(), self.column_types()):
            print(column, data_type)
            # String columns will use the value counts to compute outliers
            if data_type == "string":
                # Skip columns with too many unique values
                if len(value_count_info[column]) >= 20:
                    self.log.warning(f"Skipping column {column} too many unique values")
                    continue
                # Skip columns with a bunch of small values
                if not any(value > num_rows / 20 for value in value_count_info[column].values()):
                    self.log.warning(f"Skipping column {column} too many small values")
                    continue
                for value, count in value_count_info[column].items():
                    if count < outlier_count:
                        self.log.info(f"Found outlier feature {value} for column {column}")
                        query = f"SELECT * from {self.table_name} where {column} = '{value}' limit 5"
                        print(query)
                        df = self.query(query)
                        df["cluster"] = cluster
                        cluster += 1
                        outlier_df_list.append(df)

            elif data_type in ["bigint", "double", "int", "smallint", "tinyint"]:
                iqr = quartiles[column]["q3"] - quartiles[column]["q1"]

                # Catch cases where IQR is 0
                if iqr == 0:
                    self.log.info(f"IQR is 0 for column {column}, skipping...")
                    continue

                # Compute dataframes for the lower and upper bounds
                lower_bound = quartiles[column]["q1"] - (iqr * scale)
                upper_bound = quartiles[column]["q3"] + (iqr * scale)
                lower_df, upper_df = self._outlier_dfs(column, lower_bound, upper_bound)

                # If we have outliers, add them to the list
                for df in [lower_df, upper_df]:
                    if df is not None:
                        # Add the cluster identifier
                        df["cluster"] = cluster
                        cluster += 1
                        outlier_df_list.append(df)
                        outlier_features.append(column)

        # Combine all the outlier DataFrames, drop duplicates, and limit to 100 rows
        if outlier_df_list:
            outlier_df = pd.concat(outlier_df_list).drop_duplicates().head(100)
        else:
            self.log.warning("No outliers found for this DataSource, returning empty DataFrame")
            outlier_df = pd.DataFrame(columns=self.column_names() + ["cluster"])

        # Store the sample_df in our SageWorks metadata
        rows_json = outlier_df.to_json(orient="records", lines=True)
        self.upsert_sageworks_meta({"sageworks_outliers": rows_json})

        # Return the outlier dataframe
        return outlier_df

    def outliers_plus_samples(self) -> pd.DataFrame:
        """Give back both outliers AND 100 samples from the DataSource
        Returns:
            pd.DataFrame: A DataFrame of outliers + samples from this DataSource
        """
        outlier_df = self.outliers()
        sample_df = self.sample_df()
        sample_df["cluster"] = -1
        return pd.concat([outlier_df, sample_df], ignore_index=True).drop_duplicates()

    def value_counts(self, recompute: bool = False) -> dict[dict]:
        """Compute 'value_counts' for all the string columns in a DataSource
        Args:
            recompute(bool): Recompute the value counts (default: False)
        Returns:
            dict(dict): A dictionary of value counts for each column in the form
                 {'col1': {'value_1': X, 'value_2': Y, 'value_3': Z,...},
                  'col2': ...}
        """

        # First check if we have already computed the quartiles
        if self.sageworks_meta().get("sageworks_value_counts") and not recompute:
            return json.loads(self.sageworks_meta()["sageworks_value_counts"])

        # For every column in the table that is string, compute the value_counts
        self.log.info("Computing 'value_counts' for all string columns...")
        value_count_dict = dict()
        for column, data_type in zip(self.column_names(), self.column_types()):
            print(column, data_type)
            if data_type == "string":
                # Top value counts for this column
                query = (
                    f'SELECT "{column}", count(*) as count '
                    f"FROM {self.table_name} "
                    f'GROUP BY "{column}" ORDER BY count DESC limit 10'
                )
                top_df = self.query(query)

                # Bottom value counts for this column
                query = (
                    f'SELECT "{column}", count(*) as count '
                    f"FROM {self.table_name} "
                    f'GROUP BY "{column}" ORDER BY count ASC limit 10'
                )
                bottom_df = self.query(query).iloc[::-1]  # Reverse the DataFrame

                # Add the top and bottom value counts together
                result_df = pd.concat([top_df, bottom_df], ignore_index=True).drop_duplicates()

                # Convert int64 to int so that we can serialize to JSON
                result_df["count"] = result_df["count"].astype(int)

                # Convert any NA values to 'NaN' so that we can serialize to JSON
                result_df.fillna("NaN", inplace=True)

                # Convert the result_df into a dictionary
                value_count_dict[column] = dict(zip(result_df[column], result_df["count"]))

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
        if not self.check():
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
    from pprint import pprint

    # Retrieve a Data Source
    my_data = AthenaSource("abalone_data")

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

    # Get the input for this Artifact
    print(f"Input: {my_data.get_input()}")

    # Get Tags associated with this Artifact
    print(f"Tags: {my_data.sageworks_tags()}")

    # Get a sample of the data
    my_df = my_data.sample_df()
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

    # Get quartiles for numeric columns
    quartile_info = my_data.quartiles()
    print("\nQuartiles")
    pprint(quartile_info)

    # Get value_counts for string columns
    value_count_info = my_data.value_counts()
    print("\nValue Counts")
    pprint(value_count_info)

    # Get outliers for numeric columns
    my_outlier_df = my_data.outliers(recompute=True)
    print("\nOutliers")
    print(my_outlier_df)

    # Get ALL the AWS Metadata associated with this Artifact
    print("\n\nALL Meta")
    pprint(my_data.aws_meta())

    # Test an Data Source that doesn't exist
    print("\n\nTesting a Data Source that does not exist...")
    my_data = AthenaSource("does_not_exist")
    assert not my_data.check()
    my_data.sageworks_meta()

    # Now delete the AWS artifacts associated with this DataSource
    # print('Deleting SageWorks Data Source...')
    # my_data.delete()
