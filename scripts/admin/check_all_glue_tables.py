import awswrangler as wr
from workbench.api import Meta
from pprint import pprint
from datetime import datetime, timedelta, timezone

older_than = datetime.now(timezone.utc) - timedelta(days=90)


def glue_table_inspection(database_scope):
    """Delete invalid views from the AWS Data Catalog"""

    # Create an instance of the Meta class
    meta = Meta()

    # Get all the data sources
    """
    ds_tables = list(meta.data_sources()["Name"].values)
    print(ds_tables)

    # Find all views/supplemental data tables for the data sources
    for ds_table in ds_tables:
        view_tables = list_view_tables(ds_table, database="workbench")
        supp_data_tables = list_supplemental_data_tables(ds_table, database="workbench")
        print(view_tables)
        print(supp_data_tables)

    # Find all feature sets and their tables
    feature_sets = list(meta.feature_sets()["Feature Group"].values)
    fs_tables = [FeatureSet(fs_name).data_source.table for fs_name in feature_sets]
    print(fs_tables)
    """

    # Grab the boto3 session from the DataCatalog
    boto3_session = meta.boto3_session

    # Okay now delete invalid views for each database
    for database in database_scope:
        # Get all the glue catalog tables/views in this database
        print("*" * 50)
        print(f"Checking database: {database}...")
        print("*" * 50)
        all_tables = list(wr.catalog.get_tables(database=database, boto3_session=boto3_session))

        # Sort the tables by UpdateTime and filter out the newer tables
        all_tables.sort(key=lambda x: x["UpdateTime"])
        older_tables = [table for table in all_tables if table["UpdateTime"] < older_than]

        # Pull some specific data fields
        fields = ["Name", "TableType", "UpdateTime"]
        for table in older_tables:
            table_data = {}
            for field in fields:
                value = table[field]
                if isinstance(value, datetime):
                    value = value.strftime("%Y-%m-%d")
                table_data[field] = value
            pprint(table_data)

            # Ask if user wants to delete the table
            delete_table = input("Delete this table? (y/n): ")
            if delete_table.lower() == "y":
                wr.catalog.delete_table_if_exists(database, table["Name"], boto3_session=boto3_session)
                print(f"Deleted table {table['Name']}")
            print()


if __name__ == "__main__":
    database_scope = ["workbench", "sagemaker_featurestore"]
    glue_table_inspection(database_scope)
