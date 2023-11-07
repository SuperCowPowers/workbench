from sageworks.aws_service_broker.aws_account_clamp import AWSAccountClamp
from datetime import datetime

# Initialize a session using your AWS credentials
aws_account_clamp = AWSAccountClamp()
session = aws_account_clamp.boto_session()

glue_client = session.client("glue")

# Specify your source and destination databases
source_database = "sagemaker_featurestore"
destination_database = "sagemaker_featurestore_copy"

# Create the destination database if it doesn't exist
aws_account_clamp.ensure_aws_catalog_db(destination_database)


def copy_tables(src_db, dest_db):
    # Get the list of tables from the source database
    paginator = glue_client.get_paginator("get_tables")
    for page in paginator.paginate(DatabaseName=src_db):
        for table in page["TableList"]:
            # Construct the TableInput
            table_input = {
                "Name": table["Name"],
                "Description": table.get("Description", ""),
                "Retention": table.get("Retention", 0),
                "StorageDescriptor": table["StorageDescriptor"],
                "PartitionKeys": table.get("PartitionKeys", []),
                "TableType": table.get("TableType", ""),
                "Parameters": table.get("Parameters", {}),
            }

            # The 'ViewOriginalText' and 'ViewExpandedText' fields are for views
            if "ViewOriginalText" in table:
                table_input["ViewOriginalText"] = table["ViewOriginalText"]
            if "ViewExpandedText" in table:
                table_input["ViewExpandedText"] = table["ViewExpandedText"]

            try:
                # Create the table in the destination database
                glue_client.create_table(DatabaseName=dest_db, TableInput=table_input)
                print(f"Table {table['Name']} copied successfully.")
            except glue_client.exceptions.AlreadyExistsException:
                print(f"Table {table['Name']} already exists in the destination database.")
            except Exception as e:
                print(f"Error copying table {table['Name']}: {e}")


# Run the function to start copying tables
copy_tables(source_database, destination_database)
