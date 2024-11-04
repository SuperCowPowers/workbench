# SageWorks Imports
from sageworks.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

boto3_session = AWSAccountClamp().boto3_session


# Create a Glue client
glue_client = boto3_session.client("glue")

# Define the database and table names
database_name = "sageworks"
table_name = "test_table"

# Define the original column names with mixed case
original_column_dict = {"Id": "int", "Name": "string", "Age": "int"}


# Delete the table if it already exists
try:
    glue_client.delete_table(DatabaseName=database_name, Name=table_name)
except glue_client.exceptions.EntityNotFoundException:
    pass

# Create the Glue table
glue_client.create_table(
    DatabaseName=database_name,
    TableInput={
        "Name": table_name,
        "StorageDescriptor": {
            "Columns": [{"Name": name, "Type": dtype} for name, dtype in original_column_dict.items()],
            "Location": "s3://test-bucket/test_table/",
            "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "SerdeInfo": {"SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"},
        },
        "TableType": "EXTERNAL_TABLE",
    },
)

# Read back the Glue table
response = glue_client.get_table(DatabaseName=database_name, Name=table_name)

# Extract the column names from the response
retrieved_columns = [col["Name"] for col in response["Table"]["StorageDescriptor"]["Columns"]]

# Compare the original column names to the retrieved column names
original_columns = list(original_column_dict.keys())
print("\nOriginal Columns")
print(original_columns)
print("\nRetrieved Columns")
print(retrieved_columns)
