import argparse
from workbench.api import DataSource
from workbench.core.artifacts.athena_source import AthenaSource
from workbench.core.transforms.pandas_transforms import PandasToData


def migrate(name):
    old_database = "sageworks"

    # Get the old datasource and grab a full dataframe
    print(f"Fetching data from old DataSource: {name} in database: {old_database}")
    old_ds = AthenaSource(name, database=old_database)
    old_df = old_ds.query(f"SELECT * FROM {old_ds.table}")

    # Create the new DataSource (defaults to workbench database)
    print(f"Migrating DataSource {name} to the new database...")
    df_to_data = PandasToData(name)  # Use the same name for the new DataSource
    df_to_data.set_output_tags(old_ds.get_tags())  # Transfer tags from the old DataSource
    df_to_data.set_input(old_df)  # Pass the data from the old DataSource
    df_to_data.transform()  # Transform and save the new DataSource

    # Validate that the new DataSource has the same data as the old one
    print(f"Validating the new DataSource {name}...")
    new_ds = DataSource(name)
    new_df = new_ds.pull_dataframe()

    if old_df.equals(new_df):
        print("Migration Successful!")
        print(f"Deleting old DataSource: {name} from database: {old_database}")
        old_ds.delete()
    else:
        print("Migration Failed! The new DataSource does not match the old one.")


if __name__ == "__main__":
    # Argument parsing for the DataSource name
    parser = argparse.ArgumentParser(description="Migrate a DataSource from the old database to the new one.")
    parser.add_argument(
        "name", 
        type=str, 
        help="The name of the DataSource to migrate"
    )
    args = parser.parse_args()

    # Perform the migration
    migrate(args.name)

