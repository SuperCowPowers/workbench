"""Class: DataCatalog"""
import sys
import argparse
import awswrangler as wr
import json

# SageWorks Imports
from sageworks.aws_service_broker.aws_service_connectors.connector import Connector


class DataCatalog(Connector):
    """DataCatalog: Helper Class for the AWS Data Catalog"""
    def __init__(self, database_scope: (str, list) = 'sageworks'):
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

    def refresh(self):
        """Load/reload all the tables in all the catalog databases"""

        # For each database in our scoped list, load the tables
        for database in self.scoped_database_list:
            print(f"Reading Data Catalog Database: {database}...")
            table_list = wr.catalog.get_tables(database=database, boto3_session=self.boto_session)

            # Convert to a data structure with direct lookup
            self.data_catalog_metadata[database] = {table['Name']: table for table in table_list}

    def metadata(self) -> dict:
        """Get all the table information in this database"""
        return self.data_catalog_metadata

    def get_scoped_database_list(self):
        """Return a list of databases within the defined scope for this class"""
        all_databases = [db['Name'] for db in wr.catalog.get_databases(boto3_session=self.boto_session)]
        if self.database_scope == ['all']:
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

    def get_table_tags(self, database: str, table_name: str) -> list:
        """Get the table tag list for the given table name"""
        table = self.get_table(database, table_name)
        return json.loads(table['Parameters'].get('tags', '[]'))

    def set_table_tags(self, database: str, table_name: str, tags: list):
        """Set the tags for a specific table"""
        wr.catalog.upsert_table_parameters(parameters={'tags': json.dumps(tags)},
                                           database=database,
                                           table=table_name, boto3_session=self.boto_session)

    def add_table_tags(self, database: str, table_name: str, tags: list):
        """Add some the tags for a specific table"""
        current_tags = json.loads(wr.catalog.get_table_parameters(database, table_name).get('tags'))
        new_tags = list(set(current_tags).union(set(tags)))
        wr.catalog.upsert_table_parameters(parameters={'tags': json.dumps(new_tags)},
                                           database=database,
                                           table=table_name, boto3_session=self.boto_session)


if __name__ == '__main__':
    from pprint import pprint

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print('Unrecognized args: %s' % commands)
        sys.exit(1)

    # Create the class and get the AWS Data Catalog database info
    catalog = DataCatalog()

    # The connectors need an explicit refresh to populate themselves
    catalog.refresh()

    # List databases and tables
    for my_database in catalog.get_database_names():
        print(f"{my_database}")
        for name in catalog.get_table_names(my_database):
            print(f"\t{name}")

    # Get a specific table
    my_database = 'sageworks'
    my_table = 'aqsol_data'
    table_info = catalog.get_table(my_database, my_table)
    pprint(table_info)

    # Get the tags for this table
    my_tags = catalog.get_table_tags(my_database, my_table)
    print(f"Tags: {my_tags}")

    # Set the tags for this table
    catalog.set_table_tags(my_database, my_table, ['public', 'solubility'])

    # Refresh the connector to get the latest info from AWS Data Catalog
    catalog.refresh()

    # Get the tags for this table
    my_tags = catalog.get_table_tags(my_database, my_table)
    print(f"Tags: {my_tags}")

    # Set the tags for this table
    catalog.add_table_tags(my_database, my_table, ['aqsol', 'smiles'])

    # Refresh the connector to get the latest info from AWS Data Catalog
    catalog.refresh()

    # Get the tags for this table
    my_tags = catalog.get_table_tags(my_database, my_table)
    print(f"Tags: {my_tags}")
