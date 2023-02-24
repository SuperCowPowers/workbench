"""Class: DataCatalogDatabase"""
import sys
import argparse
import awswrangler as wr
import json


class DataCatalogDatabase:
    """DataCatalogDatabase: Helper Class for the AWS Data Catalog Database"""
    def __init__(self, database: str):
        # Store our database name
        self.database = database
        self.table_list = None
        self.table_lookup = None

        # Load in the tables from the database
        self.reload_tables()

    def reload_tables(self):
        """Load/reload the tables in the database"""
        # Grab all the tables in this database
        print(f"Reading Data Catalog Database: {self.database}...")
        self.table_list = wr.catalog.get_tables(database=self.database)

        # Convert to a data structure with direct lookup
        self.table_lookup = {table['Name']: table for table in self.table_list}

    def get_table_names(self) -> list:
        """Get all the table names in this database"""
        return list(self.table_lookup.keys())

    def get_table(self, table_name) -> dict:
        """Get the table information for the given table name"""
        return self.table_lookup.get(table_name)

    def get_table_tags(self, table_name) -> list:
        """Get the table tag list for the given table name"""
        table = self.get_table(table_name)
        return json.loads(table['Parameters'].get('tags', '[]'))

    def set_table_tags(self, table_name: str, tags: list):
        """Set the tags for a specific table"""
        wr.catalog.upsert_table_parameters(parameters={'tags': json.dumps(tags)},
                                           database=self.database,
                                           table=table_name)

    def add_table_tags(self, table_name: str, tags: list):
        """Add some the tags for a specific table"""
        current_tags = json.loads(wr.catalog.get_table_parameters(self.database, table_name).get('tags'))
        new_tags = list(set(current_tags).union(set(tags)))
        wr.catalog.upsert_table_parameters(parameters={'tags': json.dumps(new_tags)},
                                           database=self.database,
                                           table=table_name)


if __name__ == '__main__':
    from pprint import pprint

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='sageworks', help='AWS Data Catalog Database')
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print('Unrecognized args: %s' % commands)
        sys.exit(1)

    # Create the class and get the AWS Data Catalog database info
    cat_db = DataCatalogDatabase(args.database)

    # List tables in the database
    print(f"{cat_db.database}")
    for name in cat_db.get_table_names():
        print(f"\t{name}")

    # Get a specific table
    my_table = 'aqsol_data'
    table_info = cat_db.get_table(my_table)
    pprint(table_info)

    # Get the tags for this table
    tags = cat_db.get_table_tags(my_table)
    print(f"Tags: {tags}")

    # Set the tags for this table
    cat_db.set_table_tags(my_table, ['public', 'solubility'])

    # Reload tables
    cat_db.reload_tables()

    # Get the tags for this table
    tags = cat_db.get_table_tags(my_table)
    print(f"Tags: {tags}")

    # Set the tags for this table
    cat_db.add_table_tags(my_table, ['aqsol', 'smiles'])

    # Reload tables
    cat_db.reload_tables()

    # Get the tags for this table
    tags = cat_db.get_table_tags(my_table)
    print(f"Tags: {tags}")
