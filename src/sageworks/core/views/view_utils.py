"""View Utils: A set of utilities for creating and managing views in SageWorks"""

import logging
import pandas as pd
import awswrangler as wr

# SageWorks Imports
from sageworks.api import DataSource

log = logging.getLogger("sageworks")


# Get a list of columns from an Athena table/view
def get_column_list(data_source: DataSource, source_table: str) -> list[str]:
    """Get a list of columns from an Athena table/view

    Args:
        data_source (DataSource): The DataSource object
        source_table (str): The table/view name

    Returns:
        list[str]: A list of column names
    """

    # Query to get the column names
    column_query = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = '{data_source.get_database()}' AND table_name = '{source_table}'
    """
    df = data_source.query(column_query)
    return df["column_name"].tolist()


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
    boto_session = data_source.boto_session
    s3_path = f"s3://{bucket}/data-quality/{table_name}/"

    # Store the DataFrame as a Glue Catalog Table
    wr.s3.to_parquet(
        df=df,
        path=s3_path,
        dataset=True,
        mode="overwrite",
        database=database,
        table=table_name,
        boto3_session=boto_session,
    )

    # Verify that the table is created
    glue_client = boto_session.client("glue")
    try:
        glue_client.get_table(DatabaseName=database, Name=table_name)
        log.info(f"Table {table_name} successfully created in database {database}.")
    except glue_client.exceptions.EntityNotFoundException:
        log.critical(f"Failed to create table {table_name} in database {database}.")
