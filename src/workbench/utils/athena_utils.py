import os
import logging
import boto3
import time
import pandas as pd
import awswrangler as wr

# Workbench imports
from workbench.utils.performance_utils import performance
from workbench.core.cloud_platform.aws.boto_session import get_boto3_session
from workbench.core.parameter_store_core import ParameterStoreCore as ParameterStore

log = logging.getLogger("workbench")


def table_s3_path(database: str, table_name: str) -> str:
    """Get the S3 path for a Glue Catalog Table

    Args:
        database (str): The name of the Glue Catalog database
        table_name (str): The name of the table

    Returns:
        str: The S3 path for the table
    """
    # Get the Workbench Bucket
    param_key = "/workbench/config/workbench_bucket"
    workbench_bucket = ParameterStore().get(param_key)
    if workbench_bucket is None:
        # Try to get from environment variable as fallback
        workbench_bucket = os.environ.get("WORKBENCH_BUCKET")
        if workbench_bucket is None:
            raise ValueError(f"Set '{param_key}' in Parameter Store or set WORKBENCH_BUCKET ENV variable.")
        else:
            log.info(f"Upserting WORKBENCH_BUCKET={workbench_bucket} into Parameter Store at '{param_key}'")
            ParameterStore().upsert(param_key, workbench_bucket)

    # Return the S3 path for the table
    return f"s3://{workbench_bucket}/athena/{database}/{table_name}/"


def dataframe_to_table(df: pd.DataFrame, database: str, table_name: str, mode: str = "append"):
    """Store a DataFrame as a Glue Catalog Table

    Args:
        df (pd.DataFrame): The DataFrame to store
        database (str): The name of the Glue Catalog database
        table_name (str): The name of the table to store
        mode (str): The mode to use when storing the DataFrame (default: "append")
    """
    log.info("Assuming Workbench Execution Role...")
    boto3_session = get_boto3_session()

    # Get the Workbench Bucket
    param_key = "/workbench/config/workbench_bucket"
    workbench_bucket = ParameterStore().get(param_key)
    if workbench_bucket is None:
        raise ValueError(f"Set '{param_key}' in Parameter Store.")

    # Create the S3 path
    s3_path = table_s3_path(database, table_name)

    # Convert timestamp columns to UTC
    for col in df.columns:
        if df[col].dtype.name.startswith("datetime"):
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize("UTC")
            else:
                df[col] = df[col].dt.tz_convert("UTC")

    # Store the DataFrame as a Glue Catalog Table
    wr.s3.to_parquet(
        df=df,
        path=s3_path,
        dataset=True,
        mode=mode,
        schema_evolution=False,
        database=database,
        table=table_name,
        boto3_session=boto3_session,
    )


def delete_table(table_name: str, database: str, include_s3_files: bool = True):
    """Delete a table from the Glue Catalog

    Args:
        table_name (str): The name of the table to delete
        database (str): The name of the database containing the table
        include_s3_files (bool): Whether to delete the S3 files associated with the table
    """
    log.info("Assuming Workbench Execution Role...")
    boto3_session = get_boto3_session()

    # Get the Workbench Bucket
    param_key = "/workbench/config/workbench_bucket"
    workbench_bucket = ParameterStore().get(param_key)
    if workbench_bucket is None:
        raise ValueError(f"Set '{param_key}' in Parameter Store.")

    # Create the S3 path
    s3_path = f"s3://{workbench_bucket}/athena/{database}/{table_name}/"

    # Delete the table
    wr.catalog.delete_table_if_exists(database=database, table=table_name, boto3_session=boto3_session)

    # Verify that the table is deleted
    glue_client = boto3_session.client("glue")
    try:
        glue_client.get_table(DatabaseName=database, Name=table_name)
        log.error(f"Failed to delete table {table_name} in database {database}.")
    except glue_client.exceptions.EntityNotFoundException:
        log.info(f"Table {table_name} successfully deleted from database {database}.")

    # Delete the S3 files if requested
    if include_s3_files:
        log.info(f"Deleting S3 files at {s3_path}...")
        wr.s3.delete_objects(s3_path, boto3_session=boto3_session)
        log.info(f"S3 files at {s3_path} deleted.")


@performance
def compute_athena_table_hash(database: str, table: str, session: boto3.session, s3_scratch: str) -> str:
    """
    Compute a hash of an Athena table by concatenating all rows and columns.

    Args:
        database (str): The database name
        table (str): The table name
        session (boto3.session.Session): The boto3 session
        s3_scratch (str): S3 bucket and prefix for Athena query results
    Returns:
        str: MD5 hash of the table content
    """
    athena = session.client("athena")
    s3 = session.client("s3")

    def check_query_status(execution_id):
        """Check the status of an Athena query and return detailed failure info if applicable."""
        query_execution = athena.get_query_execution(QueryExecutionId=execution_id)
        status = query_execution["QueryExecution"]["Status"]
        state = status["State"]

        if state == "FAILED":
            reason = status.get("StateChangeReason", "No additional details provided.")
            print(f"Query failed. Reason: {reason}")
            raise RuntimeError(f"Athena query failed with reason: {reason}")
        elif state == "CANCELLED":
            print("Query was cancelled.")
            raise RuntimeError("Athena query was cancelled.")
        return state

    # Step 1: Retrieve the schema for the table
    schema_query = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = '{database}' AND table_name = '{table}'
    ORDER BY ordinal_position
    """

    schema_execution_id = athena.start_query_execution(
        QueryString=schema_query,
        QueryExecutionContext={"Database": "information_schema"},
        ResultConfiguration={"OutputLocation": s3_scratch},
    )["QueryExecutionId"]

    # Wait for schema query to complete
    while True:
        state = check_query_status(schema_execution_id)
        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(1)

    # Retrieve schema query results
    schema_result = athena.get_query_results(QueryExecutionId=schema_execution_id)
    columns = [row["Data"][0]["VarCharValue"] for row in schema_result["ResultSet"]["Rows"][1:]]  # Skip header

    # Step 2: Build the hash query dynamically
    column_concat = ", '|', ".join([f"CAST({col} AS VARCHAR)" for col in columns])
    hash_query = f"""
    SELECT MD5(
        CAST(
            array_join(
                array_agg(
                    CONCAT({column_concat})
                ),
                '|'
            ) AS varbinary
        )
    ) AS table_hash
    FROM {database}.{table}
    """

    hash_execution_id = athena.start_query_execution(
        QueryString=hash_query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": s3_scratch},
    )["QueryExecutionId"]

    # Wait for hash query to complete
    while True:
        state = check_query_status(hash_execution_id)
        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(1)

    # Retrieve the S3 location of the hash query result
    query_execution = athena.get_query_execution(QueryExecutionId=hash_execution_id)
    result_location = query_execution["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
    bucket, key = result_location.replace("s3://", "").split("/", 1)

    # Retrieve hash query result from S3
    obj = s3.get_object(Bucket=bucket, Key=key)
    table_hash = obj["Body"].read().decode("utf-8").splitlines()[1]  # Skip header row
    table_hash = table_hash.replace(" ", "").replace('"', "")  # Remove spaces and quotes from the hash

    # Clean up the temporary result files
    wr.s3.delete_objects(path=s3_scratch)

    return table_hash


if __name__ == "__main__":
    # Test the athena utils functions
    from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
    from workbench.utils.config_manager import ConfigManager

    # Get our boto3 session
    session = AWSAccountClamp().boto3_session

    # Get our workbench bucket
    cm = ConfigManager()
    workbench_bucket = cm.get_config("WORKBENCH_BUCKET")

    # Setup a temporary S3 prefix for the Athena output
    s3_scratch = f"s3://{workbench_bucket}/temp/athena_output"

    # Compute the hash for a table
    database_name = "workbench"
    table_name = "abalone_data"
    table_hash = compute_athena_table_hash(database_name, table_name, session, s3_scratch)
    print(f"Hash for {database_name}.{table_name}: {table_hash}")

    # Remove the temporary S3 prefix
    s3 = session.client("s3")
    s3.delete_object(Bucket=workbench_bucket, Key=f"temp/athena_output/{table_hash}.csv")
