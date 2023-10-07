"""Class: DataCatalog"""
import sys
import argparse
import awswrangler as wr

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.connector import Connector


class DataCatalog(Connector):
    """DataCatalog: Helper Class for the AWS Data Catalog"""

    def __init__(self, database_scope: (str, list) = "sageworks"):
        """DataCatalog: Helper Class for the AWS Data Catalog
        Args: -
            database_scope (str, list) - The scope of the databases to query (default = 'sageworks',
                                         can be list ['a', 'b'], also 'all' for account wide scope)
        """
        # Call SuperClass Initialization
        super().__init__()

        # Set up our database scope
        self.database_scope = database_scope if isinstance(database_scope, list) else [database_scope]
        self.scoped_database_list = self.get_scoped_database_list()

        # Set up our internal data storage
        self.data_catalog_metadata = {}

    def check(self) -> bool:
        """Check if we can reach/connect to this AWS Service"""
        try:
            wr.catalog.get_databases(boto3_session=self.boto_session)
            return True
        except Exception as e:
            self.log.critical(f"Error connecting to AWS Data Catalog: {e}")
            return False

    def refresh_impl(self):
        """Load/reload all the tables in all the catalog databases"""

        # For each database in our scoped list, load the tables
        for database in self.scoped_database_list:
            self.log.debug(f"Reading Data Catalog Database: {database}...")
            table_list = wr.catalog.get_tables(database=database, boto3_session=self.boto_session)

            # Convert to a data structure with direct lookup
            self.data_catalog_metadata[database] = {table["Name"]: table for table in table_list}

    def aws_meta(self) -> dict:
        """Return ALL the AWS metadata for this AWS Service"""
        return self.data_catalog_metadata

    def get_scoped_database_list(self):
        """Return a list of databases within the defined scope for this class"""
        all_databases = [db["Name"] for db in wr.catalog.get_databases(boto3_session=self.boto_session)]
        if self.database_scope == ["all"]:
            return all_databases

        # Return a subset of the databases
        return list(set(all_databases).intersection(set(self.database_scope)))

    def get_database_names(self) -> list:
        """Get all the database names from AWS Data Catalog"""
        return list(self.data_catalog_metadata.keys())

    def get_table_names(self, database: str) -> list:
        """Get all the table names in this database"""
        return list(self.data_catalog_metadata[database].keys())

    def get_table(self, database: str, table_name: str) -> dict:
        """Get the table information for the given table name"""
        return self.data_catalog_metadata[database].get(table_name)


if __name__ == "__main__":
    from pprint import pprint

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print("Unrecognized args: %s" % commands)
        sys.exit(1)

    # Create the class and get the AWS Data Catalog database info
    catalog = DataCatalog()

    # The connectors need an explicit refresh to populate themselves
    catalog.refresh()

    # Get ALL the AWS metadata for the AWS Data Catalog
    pprint(catalog.aws_meta())

    # List databases and tables
    for my_database in catalog.get_database_names():
        print(f"{my_database}")
        for name in catalog.get_table_names(my_database):
            print(f"\t{name}")

    # Get a specific table
    my_database = "sageworks"
    my_table = "test_data"
    table_info = catalog.get_table(my_database, my_table)
    pprint(table_info)
