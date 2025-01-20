from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
import time
from botocore.exceptions import ClientError
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize a session using your AWS credentials
aws_account_clamp = AWSAccountClamp()
session = aws_account_clamp.boto3_session
glue_client = session.client("glue")

# Specify your source and destination databases
source_database = "sageworks"
destination_database = "workbench"


def ensure_database_exists(db_name):
    try:
        glue_client.get_database(Name=db_name)
        logger.info(f"Database {db_name} already exists.")
    except glue_client.exceptions.EntityNotFoundException:
        glue_client.create_database(DatabaseInput={"Name": db_name})
        logger.info(f"Database {db_name} created successfully.")


def sanitize_storage_descriptor(storage_descriptor):
    unsupported_fields = ["ColumnsStatistics", "SortColumns"]  # Add more if necessary
    for field in unsupported_fields:
        storage_descriptor.pop(field, None)
    return storage_descriptor


def create_table_with_retry(dest_db, table_input, retries=3, delay=5):
    for attempt in range(retries):
        try:
            glue_client.create_table(DatabaseName=dest_db, TableInput=table_input)
            return True
        except ClientError as e:
            if attempt < retries - 1:
                logger.warning(f"Retrying table {table_input['Name']}... ({attempt+1}/{retries})")
                time.sleep(delay)
            else:
                raise e


def flip_parameters(parameters: dict) -> dict:
    return {key.replace("sageworks", "workbench"): value for key, value in parameters.items()}


def copy_tables(src_db, dest_db):
    # Ensure destination database exists
    ensure_database_exists(dest_db)

    # Get the list of tables from the source database
    paginator = glue_client.get_paginator("get_tables")
    for page in paginator.paginate(DatabaseName=src_db):
        for table in page["TableList"]:
            # Flip 'sageworks' to 'workbench' in parameters
            updated_parameters = flip_parameters(table.get("Parameters", {}))

            # Construct the TableInput
            table_input = {
                "Name": table["Name"],
                "Description": table.get("Description", ""),
                "Retention": table.get("Retention", 0),
                "StorageDescriptor": sanitize_storage_descriptor(table["StorageDescriptor"]),
                "PartitionKeys": table.get("PartitionKeys", []),
                "TableType": table.get("TableType", ""),
                "Parameters": updated_parameters,
            }

            # Add view-specific fields
            if "ViewOriginalText" in table:
                table_input["ViewOriginalText"] = table["ViewOriginalText"]
            if "ViewExpandedText" in table:
                table_input["ViewExpandedText"] = table["ViewExpandedText"]

            try:
                create_table_with_retry(dest_db, table_input)
                logger.info(f"Table {table['Name']} copied successfully.")
            except glue_client.exceptions.AlreadyExistsException:
                logger.warning(f"Table {table['Name']} already exists in the destination database.")
            except Exception as e:
                logger.error(f"Error copying table {table['Name']}: {e}")


# Run the function to start copying tables
copy_tables(source_database, destination_database)
