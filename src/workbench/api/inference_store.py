"""InferenceStore: Manage Inference Results using AWS S3/Parquet/Snappy with Athena Queries"""

import logging
import time

import pandas as pd
import awswrangler as wr

# Workbench Imports
from workbench.core.cloud_platform.aws.boto_session import get_boto3_session
from workbench.utils.athena_utils import (
    athena_output_s3_path,
    dataframe_to_table,
    table_s3_path,
    delete_table,
)


class InferenceStore:
    """InferenceStore: Manage Inference Results using AWS S3/Parquet/Snappy with Athena Queries

    Common Usage:
        ```python
        inf_store = InferenceStore()

        # List all Models in the Inference Store
        inf_store.models()

        # List the total number of rows in the Inference Store
        inf_store.total_rows()
        ```
    """

    def __init__(self, catalog_db: str = "inference_store", table_name: str = "inference_store"):
        """Initialize InferenceStore with S3 path and auto-register Glue/Athena table"""
        self.log = logging.getLogger("workbench")
        self.catalog_db = catalog_db
        self.table_name = table_name
        self.boto3_session = get_boto3_session()
        self.schema = ["id", "model", "pred_label", "pred_value", "tags", "meta", "timestamp"]

        # Ensure the Table exists
        if not wr.catalog.does_table_exist(self.catalog_db, self.table_name, self.boto3_session):
            self.log.warning(f"Table {self.table_name} does not exist in database {self.catalog_db}.")
            self.log.important(f"Creating table {self.table_name}...")
            self._create_empty_table()

    def add_inference_results(self, df: pd.DataFrame, schema_map: dict = None, meta_fields: list = None):
        """Add inference results to the Inference Store

        Args:
            df (pd.DataFrame): The DataFrame containing inference results.
            schema_map (dict, optional): A mapping of DataFrame columns to Inference Store schema columns.
            meta_fields (list, optional): Additional metadata fields to include in the Inference Store.
        """

        # Apply schema mapping if provided
        if schema_map:
            self.log.info(f"Applying schema mapping: {schema_map}")
            # Drop existing columns that would conflict with renamed columns
            df = df.drop(columns=[col for col in schema_map.values() if col in df.columns], errors="ignore")
            df = df.rename(columns=schema_map)

        # Check for existing meta data
        if "meta" in df.columns:
            # Check for any blanks or NaNs in the meta column
            self.log.info("Using existing 'meta' column for metadata.")
            df["meta"] = df["meta"].apply(lambda x: "{}" if pd.isna(x) or x == "" else x)
            if meta_fields:
                self.log.warning("Both 'meta' column and 'meta_fields' provided. Ignoring 'meta_fields'.")

        # Convert all meta fields to a combined JSON string and put into the 'meta' column
        elif meta_fields:
            self.log.info(f"Combining metadata fields: {meta_fields}")
            df["meta"] = df[meta_fields].apply(lambda row: row.to_json(), axis=1)
            df.drop(columns=meta_fields, inplace=True)
        else:
            df["meta"] = pd.Series("{}", index=df.index, dtype="string")

        # If "pred_label" or "pred_value" columns are missing, create them with default values
        if "pred_label" not in df.columns:
            df["pred_label"] = pd.Series([None] * len(df), dtype="string")
        if "pred_value" not in df.columns:
            df["pred_value"] = pd.Series([None] * len(df), dtype="float64")

        # Verify that we have all the schema columns
        missing_columns = set(self.schema) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"DataFrame is missing required columns: {missing_columns}. Expected schema: {self.schema}"
            )

        # Enforce proper data types for all columns
        df = self._enforce_schema_types(df)

        # Subset the DataFrame to only include the schema columns
        df = df[self.schema]

        # Add the results to the Inference Store
        self.log.info(f"Adding inference results to {self.catalog_db}.{self.table_name}")
        dataframe_to_table(df, self.catalog_db, self.table_name)
        self.log.info("Inference results added successfully.")

    def total_rows(self) -> int:
        """Return the total number of rows in the Inference Store

        Returns:
            int: The total number of rows in the Inference Store.
        """
        self.log.info(f"Retrieving total rows from {self.catalog_db}.{self.table_name}")
        df = self.query(f"SELECT COUNT(*) FROM {self.table_name}")
        return df.iloc[0, 0]

    def query(self, athena_query: str) -> pd.DataFrame:
        """Run a query against the Inference Store"""
        self.log.info(f"Running query: {athena_query}")
        start_time = time.time()

        try:
            df = wr.athena.read_sql_query(
                sql=athena_query,
                database=self.catalog_db,
                ctas_approach=False,
                boto3_session=self.boto3_session,
                s3_output=athena_output_s3_path(),
            )
            execution_time = time.time() - start_time
            self.log.info(f"Query completed in {execution_time:.2f} seconds")

            # Convert tags column from string to list
            if "tags" in df.columns:
                df["tags"] = df["tags"].str.strip("[]").str.split(", ")

            # Convert timestamp columns to UTC if they are naive
            for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
                if df[col].dt.tz is None:
                    df[col] = df[col].dt.tz_localize("UTC")
                elif df[col].dt.tz.zone != "UTC":
                    df[col] = df[col].dt.tz_convert("UTC")

            return df
        except Exception as e:
            self.log.error(f"Failed to run query: {e}")
            return pd.DataFrame()

    def delete_all_data(self):
        """Delete all data in the Inference Store"""
        self.log.info(f"Deleting all data from {self.catalog_db}.{self.table_name}")
        delete_table(self.table_name, self.catalog_db)

    def _enforce_schema_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce proper data types for all columns according to the schema

        Args:
            df (pd.DataFrame): The DataFrame to type-convert

        Returns:
            pd.DataFrame: DataFrame with properly typed columns
        """
        self.log.info("Enforcing schema types...")

        # Type enforcement mapping
        type_mapping = {
            "id": "string",
            "timestamp": "datetime64[ms]",
            "model": "string",
            "pred_label": "string",
            "pred_value": "float64",
            "tags": "object",  # Special handling for lists
            "meta": "string",
        }

        for col, dtype in type_mapping.items():
            if col in df.columns:
                try:
                    if col == "tags":
                        # Ensure tags is always a list, even if empty
                        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [] if pd.isna(x) else [str(x)])
                    elif col == "timestamp":
                        # Handle timestamp conversion more carefully
                        if not pd.api.types.is_datetime64_any_dtype(df[col]):
                            df[col] = pd.to_datetime(df[col])
                    else:
                        # Standard type conversion
                        df[col] = df[col].astype(dtype)

                except Exception as e:
                    raise ValueError(f"Failed to convert column '{col}' to type '{dtype}': {e}")

        # Return the DataFrame with enforced types
        return df

    def _create_empty_table(self):
        """Create an empty table directly using AWS Data Wrangler with proper column types"""
        self.log.important(f"Creating empty table {self.table_name} in database {self.catalog_db}")

        # Define the table schema with proper types
        table_schema = {
            "id": "string",
            "timestamp": "timestamp",
            "model": "string",
            "pred_label": "string",
            "pred_value": "double",
            "tags": "array<string>",
            "meta": "string",
        }
        s3_path = table_s3_path(database=self.catalog_db, table_name=self.table_name)
        wr.catalog.create_parquet_table(
            database=self.catalog_db,
            table=self.table_name,
            path=s3_path,
            columns_types=table_schema,
            boto3_session=self.boto3_session,
        )

    def __repr__(self):
        """Return a string representation of the InferenceStore object."""
        return f"InferenceStore(catalog_db={self.catalog_db}, table_name={self.table_name})"


if __name__ == "__main__":
    """Exercise the InferenceStore Class"""

    # Create a InferenceStore manager
    inf_store = InferenceStore()

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "compound_id": [1, 2, 3],  # Note: These will be converted to strings
            "model_name": ["model1", "model2", "model3"],
            "inference_result": [0.1, 0.2, 0.3],
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "tags": [["tag1", "tag2"], ["tag2"], ["tag1", "tag3"]],
        }
    )

    # Add inference results to the Inference Store
    schema_map = {"compound_id": "id", "model_name": "model", "inference_result": "pred_value"}
    inf_store.add_inference_results(df, schema_map=schema_map)

    # List the total rows
    print(f"Total rows in Inference Store: {inf_store.total_rows()}")

    # List all models
    print("Listing Models...")
    print(inf_store.query("SELECT distinct model FROM inference_store"))
