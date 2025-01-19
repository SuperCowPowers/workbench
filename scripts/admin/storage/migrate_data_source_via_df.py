import argparse
import numpy as np
import pandas as pd

#
from workbench.api import DataSource
from workbench.core.artifacts.athena_source import AthenaSource
from workbench.core.transforms.pandas_transforms import PandasToData
from workbench.core.cloud_platform.aws.aws_meta import AWSMeta


def compare_dataframes_with_tolerance(df1: pd.DataFrame, df2: pd.DataFrame, tolerance: float = 1e-8) -> bool:
    # Check column names
    if list(df1.columns) != list(df2.columns):
        print("Column names differ:")
        print(f"df1 columns: {list(df1.columns)}")
        print(f"df2 columns: {list(df2.columns)}")
        return False

    # Check column types
    if not all(df1.dtypes == df2.dtypes):
        print("Column types differ:")
        print("df1 types:")
        print(df1.dtypes)
        print("df2 types:")
        print(df2.dtypes)
        return False

    # Check number of rows
    if len(df1) != len(df2):
        print(f"Number of rows differ: df1 has {len(df1)} rows, df2 has {len(df2)} rows.")
        return False

    # Sort dataframes by ID columns
    id_columns = ["id", "ID", "primary_key", "unique_id"]
    sort_columns = [col for col in id_columns if col in df1.columns and col in df2.columns]

    if sort_columns:
        df1 = df1.sort_values(by=sort_columns).reset_index(drop=True)
        df2 = df2.sort_values(by=sort_columns).reset_index(drop=True)

    # Compare each column
    for column in df1.columns:
        if df1[column].dtype.kind in "f":  # Float column
            is_close = np.isclose(df1[column], df2[column], atol=tolerance)
            if not np.all(is_close):
                print(f"Differences found in column: {column}")
                return False
        else:  # Non-float column
            if not df1[column].equals(df2[column]):
                print(f"Differences found in column: {column}")
                return False

    # DataFrames are equivalent
    return True


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

    print(f"Validating the new DataSource {ds_name}...")
    new_ds = DataSource(ds_name)
    new_df = new_ds.query(f"SELECT * FROM {new_ds.table}")

    # Compare the old and new DataFrames
    if compare_dataframes_with_tolerance(old_df, new_df, tolerance=1e-8):
        print(f"Migration Successful: DataSource {ds_name} has been migrated to the new database.")
    else:
        print(f"Migration Failed: DataSource {ds_name} dataframe did not match the original.")


def migrate_all():
    old_database = "sageworks"

    # Grab the DataSource in the old database
    old_datasources = AWSMeta()._list_catalog_tables(old_database)
    for ds_name in old_datasources["Name"].values:
        print(f"Starting migration for DataSource: {ds_name}")
        try:
            # print(f"Mock migration for {ds_name}")
            migrate(ds_name)
        except Exception as e:
            print(f"Migration failed for: {ds_name}. Error: {e}")


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
