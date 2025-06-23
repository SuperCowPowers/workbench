import boto3


def get_catalog_summary():
    glue = boto3.client("glue")

    # Get all databases
    databases = []
    paginator = glue.get_paginator("get_databases")
    for page in paginator.paginate():
        databases.extend(page["DatabaseList"])

    print(f"Total databases: {len(databases)}\n")

    # For each database, count tables and views
    total_tables = 0
    total_views = 0

    for db in databases:
        db_name = db["Name"]
        tables = 0
        views = 0

        # Who created this database and when?
        created_by = db.get("CreatedBy", "Unknown")
        created_date = db.get("CreateTime", "Unknown")
        print(f"{db['Name']}: created by {created_by} on {created_date}")

        # Get all tables in this database
        table_paginator = glue.get_paginator("get_tables")
        for page in table_paginator.paginate(DatabaseName=db_name):
            for table in page["TableList"]:
                if table.get("TableType") == "VIRTUAL_VIEW":
                    views += 1
                else:
                    tables += 1

        print(f"{db_name}: {tables} tables, {views} views")
        total_tables += tables
        total_views += views

    print(f"\nGrand total: {total_tables} tables, {total_views} views")


if __name__ == "__main__":
    get_catalog_summary()
