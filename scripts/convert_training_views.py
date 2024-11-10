"""Convert training views from 0/1 to FALSE/TRUE in the specified AWS Glue database.

   Note: This script is a 'schema' change for the training views, and is a 'one-time'
       operation. The code quality here is not as important as the correctness of the
       operation and since this will only be run once for existing clients and never
       again, we don't want to sweat the details.
"""

import json
import re
import base64
import logging
import awswrangler as wr
from sageworks.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

log = logging.getLogger("sageworks")

# Initialize your AWS session and Glue client
aws_account_clamp = AWSAccountClamp()
session = aws_account_clamp.boto3_session
glue_client = session.client("glue")


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


def convert_training_views(database):
    """Convert training views from 0/1 to FALSE/TRUE in the specified AWS Glue database"""

    # Use the Glue client to get the list of tables (views) from the database
    paginator = glue_client.get_paginator("get_tables")

    for page in paginator.paginate(DatabaseName=database):
        for table in page["TableList"]:
            # Check if the table name ends with "_training" and is a view
            if table["Name"].endswith("_training") and table.get("TableType") == "VIRTUAL_VIEW":
                print(f"Checking view: {table['Name']}...")

                # Decode the 'ViewOriginalText' for the view
                view_original_text = _decode_view_sql(table.get("ViewOriginalText"))
                if view_original_text and (" THEN 0" in view_original_text or " THEN 1" in view_original_text):
                    print(f"\tConverting view: {table['Name']}...")

                    # Update the THEN and ELSE view definitions by replacing 0/1 with FALSE/TRUE
                    updated_query = view_original_text.replace(" THEN 0", " THEN FALSE").replace(
                        " THEN 1", " THEN TRUE"
                    )
                    updated_query = updated_query.replace(" ELSE 0", " ELSE FALSE").replace(" ELSE 1", " ELSE TRUE")

                    # Construct the full CREATE OR REPLACE VIEW query
                    query = f"""
                    CREATE OR REPLACE VIEW {table['Name']} AS
                    {updated_query}
                    """

                    try:
                        # Execute the query using awswrangler
                        query_execution_id = wr.athena.start_query_execution(
                            sql=query,
                            database=database,
                            boto3_session=session,
                        )
                        print(f"\tQueryExecutionId: {query_execution_id}")

                        # Wait for the query to complete
                        wr.athena.wait_query(query_execution_id=query_execution_id, boto3_session=session)
                        print(f"\tSuccessfully converted view: {table['Name']}")
                    except Exception as e:
                        print(f"\tError updating view {table['Name']}: {e}")
                else:
                    print(f"\tNo conversion needed for view: {table['Name']}")


if __name__ == "__main__":
    # Specify your database scope
    database_scope = ["sagemaker_featurestore"]
    for db in database_scope:
        convert_training_views(db)
