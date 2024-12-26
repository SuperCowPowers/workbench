import argparse
from workbench.api import DataSource
from workbench.core.artifacts.athena_source import AthenaSource
from workbench.core.transforms.pandas_transforms import PandasToData
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp


def migrate(ds_name):
    old_database = "sageworks"

    # Get the old datasource and grab a full dataframe
    print(f"Fetching data from old DataSource: {ds_name} in database: {old_database}")
    old_ds = AthenaSource(ds_name, database=old_database)
    old_df = old_ds.query(f"SELECT * FROM {old_ds.table}")

    # Create the new DataSource (defaults to workbench database)
    print(f"Migrating DataSource {ds_name} to the new database...")
    df_to_data = PandasToData(ds_name)  # Use the same name for the new DataSource
    df_to_data.set_output_tags(old_ds.get_tags())  # Transfer tags from the old DataSource
    df_to_data.set_input(old_df)  # Pass the data from the old DataSource
    df_to_data.transform()  # Transform and save the new DataSource

    # Validate that the new DataSource has the same data as the old one
    print(f"Validating the new DataSource {ds_name}...")
    new_ds = DataSource(ds_name)
    new_df = new_ds.pull_dataframe()

    if old_df.equals(new_df):
        print("Migration Successful!")
        print(f"Deleting old DataSource: {ds_name} from database: {old_database}")
        old_ds.delete()
    else:
        print("Migration Failed! The new DataSource does not match the old one.")


def migrate_all():
    old_database = "sageworks"

    # Grab an Glue Client from Workbench
    aws_account_clamp = AWSAccountClamp()
    session = aws_account_clamp.boto3_session
    glue_client = session.client("glue")

    # Get a list of all tables in the old database
    print(f"Fetching all tables from the old database: {old_database}")
    paginator = glue_client.get_paginator("get_tables")
    for page in paginator.paginate(DatabaseName=old_database):
        for table_name in page["TableList"]:
            print(f"Starting migration for table: {table_name}")
            try:
                print("Fake migration")
                # migrate(table_name)
            except Exception as e:
                print(f"Migration failed for table: {table_name}. Error: {e}")


if __name__ == "__main__":
    # Argument parsing to choose between single or all DataSource migration
    parser = argparse.ArgumentParser(description="Migrate DataSources from the old database to the new one.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Migrate all DataSources from the old database",
    )
    parser.add_argument(
        "--ds_name",
        type=str,
        help="The name of a single DataSource to migrate (optional if --all is used)",
    )
    args = parser.parse_args()

    if args.all:
        migrate_all()
    elif args.ds_name:
        migrate(args.ds_name)
    else:
        print("Please specify --all to migrate all DataSources or --ds_name for a single DataSource.")
