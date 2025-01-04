# Quote from AWS Docs: https://docs.aws.amazon.com/athena/latest/ug/tables-databases-columns-names.html
"""
Use lower case for table names and table column names in Athena

Athena accepts mixed case in DDL and DML queries, but lower cases the names when it
executes the query. For this reason, avoid using mixed case for table or column names,
and do not rely on casing alone in Athena to distinguish such names. For example,
if you use a DDL statement to create a column named Castle, the column created
will be lowercased to castle.
"""


# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

boto3_session = AWSAccountClamp().boto3_session


# Create a Glue client and Athena client
glue_client = boto3_session.client("glue")
athena_client = boto3_session.client("athena")

# Define the database, table name, and S3 path
database_name = "workbench"
table_name = "test_table"
s3_path = "s3://test-bucket/test/"

# Define the original column names with mixed case
original_column_dict = {"Id": "int", "Name": "string", "Age": "int"}

# Define the Athena DDL query to create the table
ddl_query = f"""
CREATE EXTERNAL TABLE IF NOT EXISTS {database_name}.{table_name} (
    `Id` int,
    `Name` string,
    `Age` int
)
STORED AS PARQUET
LOCATION '{s3_path}'
"""

# Execute the DDL query
execution_id = athena_client.start_query_execution(
    QueryString=ddl_query,
    QueryExecutionContext={"Database": database_name},
    ResultConfiguration={"OutputLocation": "s3://your-output-bucket/path/"},
)["QueryExecutionId"]

# Wait for the query execution to complete (you may want to add a timeout or a more sophisticated waiting mechanism)
athena_client.get_query_execution(QueryExecutionId=execution_id)

# Use Glue client to read back the table (as Athena doesn't provide a direct way to get column names)
response = glue_client.get_table(DatabaseName=database_name, Name=table_name)

# Extract the column names from the response
retrieved_columns = [col["Name"] for col in response["Table"]["StorageDescriptor"]["Columns"]]

# Compare the original column names to the retrieved column names
original_columns = list(original_column_dict.keys())
print("\nOriginal Columns")
print(original_columns)
print("\nRetrieved Columns")
print(retrieved_columns)
