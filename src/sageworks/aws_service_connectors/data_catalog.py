"""Class: DataCatalog"""
import sys
import argparse
import awswrangler as wr
import json
import logging

# Local Imports
from sageworks.utils.logging import logging_setup
from sageworks.aws_service_connectors.connector import Connector

# Set up logging
logging_setup()


class DataCatalog(Connector):
    """DataCatalog: Helper Class for the AWS Data Catalog"""
    def __init__(self):
        self.log = logging.getLogger(__name__)

        # Set up our internal data storage
        self.data_catalog_metadata = {}

    def check(self) -> bool:
        """Check if we can reach/connect to this AWS Service"""
        try:
            wr.catalog.get_databases()
            return True
        except Exception as e:
            self.log.critical(f"Error connecting to AWS Data Catalog: {e}")
            return False

    def refresh(self):
        """Load/reload all the tables in all the catalog databases"""

        # For each database, load the tables
        for database in wr.catalog.get_databases():
            print(f"Reading Data Catalog Database: {database['Name']}...")
            table_list = wr.catalog.get_tables(database=database['Name'])

            # Convert to a data structure with direct lookup
            self.data_catalog_metadata[database['Name']] = {table['Name']: table for table in table_list}

    def get_metadata(self) -> list:
        """Get all the table information in this database"""
        return self.data_catalog_metadata

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

    @staticmethod
    def set_table_tags(database: str, table_name: str, tags: list):
        """Set the tags for a specific table"""
        wr.catalog.upsert_table_parameters(parameters={'tags': json.dumps(tags)},
                                           database=database,
                                           table=table_name)

    @staticmethod
    def add_table_tags(database: str, table_name: str, tags: list):
        """Add some the tags for a specific table"""
        current_tags = json.loads(wr.catalog.get_table_parameters(database, table_name).get('tags'))
        new_tags = list(set(current_tags).union(set(tags)))
        wr.catalog.upsert_table_parameters(parameters={'tags': json.dumps(new_tags)},
                                           database=database,
                                           table=table_name)


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
    tags = catalog.get_table_tags(my_database, my_table)
    print(f"Tags: {tags}")

    # Set the tags for this table
    catalog.set_table_tags(my_database, my_table, ['public', 'solubility'])

    # Refresh the connector to get the latest info from AWS Data Catalog
    catalog.refresh()

    # Get the tags for this table
    tags = catalog.get_table_tags(my_database, my_table)
    print(f"Tags: {tags}")

    # Set the tags for this table
    catalog.add_table_tags(my_database, my_table, ['aqsol', 'smiles'])

    # Refresh the connector to get the latest info from AWS Data Catalog
    catalog.refresh()

    # Get the tags for this table
    my_tags = catalog.get_table_tags(my_database, my_table)
    print(f"Tags: {my_tags}")
