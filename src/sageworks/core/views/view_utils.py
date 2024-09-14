"""View Utils: A set of utilities for creating and managing views in SageWorks"""

import base64
import json
import re
from typing import Union
import logging
import pandas as pd
import awswrangler as wr
from botocore.exceptions import ClientError

# SageWorks Imports
from sageworks.api import DataSource

log = logging.getLogger("sageworks")


# Get a list of columns from an Athena table/view
def get_column_list(data_source: DataSource, source_table: str = None) -> list[str]:
    """Get a list of columns from an Athena table/view

    Args:
        data_source (DataSource): The DataSource object
        source_table (str, optional): The table/view to get the columns from. Defaults to None.

    Returns:
        list[str]: A list of column names
    """
    source_table = source_table if source_table else data_source.table

    # Query to get the column names
    column_query = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = '{data_source.get_database()}' AND table_name = '{source_table}'
    """
    df = data_source.query(column_query)
    return df["column_name"].tolist()


def list_views(data_source: DataSource) -> list[str]:
    """Extract the last part of the view table names in a database for a DataSource

    Args:
        data_source (DataSource): The DataSource object

    Returns:
        list[str]: A list containing only the last part of the view table names
    """
    # Get the list of view tables for this data source
    view_tables = list_view_tables(data_source)
    data_source_table = data_source.table

    # Each view will have the format: {data_table_name}_{view_name}
    return [view_table.replace(data_source_table + "_", "") for view_table in view_tables]


def list_view_tables(data_source: DataSource) -> list[str]:
    """List all the view tables in a database for a DataSource

    Args:
        data_source (DataSource): The DataSource object

    Returns:
        list[str]: A list of view table names
    """
    base_table_name = data_source.table

    # Use LIKE to match table names that start with base_table_name followed by any characters
    view_query = f"""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = '{data_source.get_database()}'
      AND table_type = 'VIEW'
      AND table_name LIKE '{base_table_name}_%'
    """
    df = data_source.query(view_query)
    return df["table_name"].tolist()


def list_supplemental_data_tables(data_source: DataSource) -> list[str]:
    """List all supplemental data tables in a database for a DataSource

    Args:
        data_source (DataSource): The DataSource object

    Returns:
        list[str]: A list of supplemental data table names
    """
    base_table_name = data_source.table

    # Use REGEXP_LIKE to match table names that start with an underscore, followed by the base_table_name,
    # followed by one underscore, and no more underscores
    supplemental_data_query = f"""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = '{data_source.get_database()}'
      AND table_type = 'BASE TABLE'
      AND table_name LIKE '{base_table_name}_%'
    """
    df = data_source.query(supplemental_data_query)
    return df["table_name"].tolist()


def dataframe_to_table(data_source: DataSource, df: pd.DataFrame, table_name: str):
    """Store a DataFrame as a Glue Catalog Table

    Args:
        data_source (DataSource): The DataSource object
        df (pd.DataFrame): The DataFrame to store
        table_name (str): The name of the table to store
    """

    # Grab information from the data_source
    bucket = data_source.sageworks_bucket
    database = data_source.get_database()
    boto3_session = data_source.boto3_session
    s3_path = f"s3://{bucket}/supplemental-data/{table_name}/"

    # Store the DataFrame as a Glue Catalog Table
    wr.s3.to_parquet(
        df=df,
        path=s3_path,
        dataset=True,
        mode="overwrite",
        database=database,
        table=table_name,
        boto3_session=boto3_session,
    )

    # Verify that the table is created
    glue_client = boto3_session.client("glue")
    try:
        glue_client.get_table(DatabaseName=database, Name=table_name)
        log.info(f"Table {table_name} successfully created in database {database}.")
    except glue_client.exceptions.EntityNotFoundException:
        log.critical(f"Failed to create table {table_name} in database {database}.")


def delete_views_and_supplemental_data(data_source: DataSource):
    """Delete all views and supplemental data in a database

    Args:
        data_source (DataSource): The DataSource object
    """
    for view_table in list_view_tables(data_source):
        delete_table(data_source, view_table)
    for supplemental_data_table in list_supplemental_data_tables(data_source):
        delete_table(data_source, supplemental_data_table)


def delete_table(data_source: DataSource, table_name: str):
    """Delete a table from the Glue Catalog

    Args:
        data_source (DataSource): The DataSource object
        table_name (str): The name of the table to delete
    """
    # Grab information from the data_source
    database = data_source.get_database()
    boto3_session = data_source.boto3_session

    # Delete the table
    wr.catalog.delete_table_if_exists(database=database, table=table_name, boto3_session=boto3_session)

    # Verify that the table is deleted
    glue_client = boto3_session.client("glue")
    try:
        glue_client.get_table(DatabaseName=database, Name=table_name)
        log.error(f"Failed to delete table {table_name} in database {database}.")
    except glue_client.exceptions.EntityNotFoundException:
        log.info(f"Table {table_name} successfully deleted from database {database}.")


def view_details(
    database: str, table: str, boto3_session
) -> (Union[list, None], Union[list, None], Union[str, None], bool):
    """Pull the column names, types, and source table for the view

    Args:
        database (str): The database name
        table (str): The view table
        boto3_session: The boto3 session

    Returns:
        Union[list, None]: The column names (returns None if the table does not exist)
        Union[list, None]: The column types (returns None if the table does not exist)
        Union[str, None]: The source table the view was created from (returns None if not found)
        bool: True if the view is a JOIN view, False otherwise
    """

    # Retrieve the table metadata
    glue_client = boto3_session.client("glue")
    try:
        response = glue_client.get_table(DatabaseName=database, Name=table)

        # Extract the column names and types from the schema
        column_names = [col["Name"] for col in response["Table"]["StorageDescriptor"]["Columns"]]
        column_types = [col["Type"] for col in response["Table"]["StorageDescriptor"]["Columns"]]

        # Check if the table is a view and extract the source table from the SQL query
        if response["Table"]["TableType"] != "VIRTUAL_VIEW":
            return column_names, column_types, response["Table"]["Name"]
        elif "ViewOriginalText" in response["Table"]:
            view_sql_encoded = response["Table"]["ViewOriginalText"]
            view_sql_decoded = _decode_view_sql(view_sql_encoded)
            source_table = _extract_source_table(view_sql_decoded)
            join_view = "JOIN" in view_sql_decoded
            return column_names, column_types, source_table, join_view
        else:
            log.error(f"Failed to extract source table from view {table}.")
            return column_names, column_types, None, False

    # Handle the case where the table does not exist
    except glue_client.exceptions.EntityNotFoundException:
        log.warning(f"Table {table} not found in database {database}.")
        return None, None, None, False
    except ClientError as e:
        log.error(f"An error occurred while retrieving table info: {e}")
        return None, None, None, False


def _decode_view_sql(encoded_sql: str) -> str:
    """Decode the base64-encoded SQL query from the view.

    Args:
        encoded_sql (str): The encoded SQL query in the ViewOriginalText.

    Returns:
        str: The decoded SQL query.
    """
    # Extract the base64-encoded content from the comment
    match = re.search(r"Presto View: ([\w=+/]+)", encoded_sql)
    if match:
        base64_sql = match.group(1)
        decoded_bytes = base64.b64decode(base64_sql)
        decoded_str = decoded_bytes.decode("utf-8")

        # Parse the decoded string as JSON to extract the SQL
        try:
            view_json = json.loads(decoded_str)
            return view_json.get("originalSql", "")
        except json.JSONDecodeError:
            log.error("Failed to parse the decoded view SQL as JSON.")
            return ""
    return ""


def _extract_source_table(view_sql: str) -> Union[str, None]:
    """Extract the source table from the SQL query of the view.

    Args:
        view_sql (str): The decoded SQL query used to create the view.

    Returns:
        Union[str, None]: The source table name if found, otherwise None.
    """
    # Use regex to find the source table in the SQL query
    match = re.search(r"FROM\s+([^\s;]+)", view_sql, re.IGNORECASE)
    table = match.group(1) if match else None

    # Special case for join queries
    if table.startswith("("):
        table = table.replace("(", "")

    # Return the source table name
    return table


if __name__ == "__main__":
    """Test the various view utilities"""
    from sageworks.api import FeatureSet

    # Create a FeatureSet
    fs = FeatureSet("abalone_features")
    my_data_source = fs.data_source

    # Test view_details
    print("View Details on the FeatureSet Table...")
    print(view_details(my_data_source.get_database(), my_data_source.table, my_data_source.boto3_session))

    print("View Details on the Training View...")
    training_view = fs.view("training")
    print(view_details(training_view.database, training_view.table, my_data_source.boto3_session))

    # Test get_column_list
    print(get_column_list(my_data_source))

    # Test get_column_list (with training view)
    training_table = fs.view("training").table
    print(get_column_list(my_data_source, training_table))

    # Test list_view_tables
    print("List Views...")
    print(list_view_tables(my_data_source))

    # Test list_views
    print("List Views...")
    print(list_views(my_data_source))

    # Test list_supplemental_data
    print("List Supplemental Data...")
    print(list_supplemental_data_tables(my_data_source))

    # Test view/supplemental data deletion
    print("Deleting Views and Supplemental Data...")
    delete_views_and_supplemental_data(my_data_source)

    # Test dataframe_to_table
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})
    dataframe_to_table(my_data_source, df, "test_table")
    delete_table(my_data_source, "test_table")
