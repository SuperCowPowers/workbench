"""Class: GlueMeta"""
import sys
import argparse
import awswrangler as wr
from awswrangler.exceptions import NoFilesFound
import json
import boto3

# SLUDGE
udm_types = {'TEXT', 'SMILES', 'NMBR', 'DATE', 'XSMI', 'CTAB'}
udm_type_to_postgres = {
    'TEXT': 'text',
    'SMILES': 'text',
    'NMBR': 'real',
    'DATE': 'date',
    'XSMI': 'text',
    'CTAB': 'text'
}


# Class: GlueMeta
class GlueMeta:
    """GlueMeta: Grab information from the AWS Glue Catalog"""
    def __init__(self, database: str, skip_ignore: bool = True):
        # Store our database name
        self.database = database

        # Create our boto3 session
        self.session = boto3.session.Session()

        # Set which tags are going to be 'sticky' (i.e. won't get overwritten/removed when a tag run occurs)
        self.sticky_tags = ['combo']

        # Keep track of files that are ignored
        self.ignored_files = []

        # Populate our local table_data with all the AWS Glue Tables from this catalog database
        print(f"Reading Glue Catalog Database: {self.database}...")
        self.table_data = self._get_table_data(skip_ignore)

    def get_table_data(self) -> dict:
        """Get tag and operation data for all the tables
                    Returns:
                      {'table1': {'location': <s3_path>, 'tags': [list], 'operations': [list]},
                       'table2': {'location': <s3_path>, 'tags': [list], 'operations': [list]}
                """
        return self.table_data

    def _get_table_data(self, skip_ignore: bool) -> dict:
        """Internal: Populate the self.table_data dictionary with data from the Glue Catalog"""
        # Get Tables Descriptions for this database
        tables = self._get_glue_tables()
        organized_data = dict()
        for table in tables:
            # Load in tags and operations
            tags = json.loads(table['Parameters'].get('tags', '[]'))
            ops = json.loads(table['Parameters'].get('operations', '[]'))
            table_meta = dict({'tags': tags, 'operations': ops})

            # Skip files that have an ignore tag
            if skip_ignore and 'ignore' in tags:
                print(f"Skipping {table['Name']} with ignore tag...")
                self.ignored_files.append(table['Name'])
                continue

            # Add the S3 file location
            table_meta['location'] = table['StorageDescriptor']['Location']

            # Add the columns names (glue computed)
            column_info_list = table['StorageDescriptor']['Columns']
            column_names = [item['Name'] for item in column_info_list]
            table_meta['glue_columns'] = column_names

            # These are GLUE column types, pulling REGISTERED column types happens on demand
            glue_computed_types = [item['Type'] for item in column_info_list]
            table_meta['glue_computed_types'] = glue_computed_types
            organized_data[table['Name']] = table_meta
        return organized_data

    def tables_with_tags(self, tags: list, exclude_list: list = None) -> dict:
        """Get all the tables and have ALL the tags specified
            Returns:
              {'table1': {'tags': [list], 'operations': [list]},
               'table2': {'tags': [list], 'operations': [list]}
        """
        tables = {name: data for name, data in self.table_data.items() if set(tags).issubset(set(data['tags']))}
        if exclude_list:
            tables = {name: data for name, data in tables.items() if not any([item in data['tags'] for item in exclude_list])}
        return tables

    def _get_glue_tables(self):
        """Internal: Get the table objects from the Glue Catalog Database"""
        return wr.catalog.get_tables(database=self.database)

    def get_table_info(self, table_name: str):
        """Get information for a specific table
           Returns:
               {'tags': <list>, 'operations': <list>} or None (if not found)
            """
        return self.table_data.get(table_name)

    def get_table_columns(self, table_name: str):
        """Get the columns for a specific table"""
        return self.table_data[table_name]['glue_columns']

    def get_column_dtypes(self, table_name: str) -> dict:
        """Lazy: Get the original columns and registered types for this table"""
        if self.has_registered_types(table_name):
            return self.table_data[table_name]['registered_types']

        # Get the original columns and the registered dtypes
        table_info = self.get_table_info(table_name)
        print('\tRegistering Types...')
        table_info['registered_types'] = self._get_registered_types(table_info['location'])
        print('\tRegistering Original Column Names...')
        table_info['original_columns'] = list(table_info['registered_types'].keys())
        dtypes = table_info['registered_types']

        # Return a dictionary of ORIGINAL column names mapped to dtypes
        return dtypes

    @staticmethod
    def get_udm_types():
        return udm_type_to_postgres

    def has_registered_types(self, table_name: str) -> bool:
        """Does this AWS Glue Catalog Table have registered types"""
        table_info = self.get_table_info(table_name)
        return True if 'registered_types' in table_info else False

    def get_original_columns(self, table_name: str) -> list:
        """On Demand Retrieval of Original Column Names"""
        table_info = self.get_table_info(table_name)
        if 'original_columns' in table_info:
            return table_info['original_columns']
        else:
            table_info['original_columns'] = self._get_original_columns(table_info['location'])
            return table_info['original_columns']

    def _get_original_columns(self, s3_file_path: str) -> list:
        """Internal: Get the Original columns names (glue will lowercase, underscore, etc. the column names)"""

        # Grab the CSV file just to get the columns headers (only grab 1 row)
        print('Getting Original Columns from S3...')
        csv_df = wr.s3.read_csv(path=s3_file_path, boto3_session=self.session, nrows=1)
        return list(csv_df.columns)

    def _get_registered_types(self, s3_file_path: str) -> dict:
        """Internal: Sometimes types will be registered by providing as associated *.txt file"""

        # Grab the TXT file (that might not exist)
        print('Getting Registered Types from S3...')
        txt_file_path = s3_file_path.replace(".csv", ".txt")
        try:
            txt_df = wr.s3.read_csv(path=txt_file_path, boto3_session=self.session)
        except NoFilesFound:
            return {}

        # The datatypes are in the first row
        dtypes = txt_df.to_dict('records')[0]

        # Lets sanity check the dtypes
        known_dtypes = {'TEXT', 'SMILES', 'NMBR', 'DATE', 'XSMI', 'CTAB'}
        if not set(dtypes.values()).issubset(known_dtypes):
            invalid_types = list(set(dtypes).difference(known_dtypes))
            print(f"Invalid datatypes {invalid_types}")
            return {}

        # Types look okay, return them
        return dtypes

    def get_table_tags(self, table_name: str):
        """Set the tags for a specific table"""
        return self.table_data[table_name]['tags']

    def set_table_tags(self, table_name: str, tags: list, push_to_aws: bool = False):
        """Set the tags for a specific table"""

        # Special logic to avoid overwriting sticky tags
        sticky_tags = set(self.sticky_tags).intersection(set(self.table_data[table_name]['tags']))
        self.table_data[table_name]['tags'] = list(set(tags).union(sticky_tags))
        if push_to_aws:
            wr.catalog.upsert_table_parameters(parameters={'tags': json.dumps(self.table_data[table_name]['tags'])},
                                               database=self.database,
                                               table=table_name)

    def add_table_tags(self, table_name: str, tags: list, push_to_aws=False):
        """Add the tags for a specific table"""
        self.table_data[table_name]['tags'] = list(set(self.table_data[table_name]['tags']).union(set(tags)))
        if push_to_aws:
            wr.catalog.upsert_table_parameters(parameters={'tags': json.dumps(self.table_data[table_name]['tags'])},
                                               database=self.database,
                                               table=table_name)

    def write_tags_to_aws(self):
        """Set the tags for a specific table"""
        for name, table_info in self.table_data.items():
            print(name, table_info['tags'])
            wr.catalog.upsert_table_parameters(parameters={'tags': json.dumps(table_info['tags'])},
                                               database=self.database,
                                               table=name)

    def get_table_ops(self, table_name: str):
        """Set the tags for a specific table"""
        return self.table_data[table_name]['operations']

    def set_table_ops(self, table_name: str, operations: list):
        """Set the operations for a specific table"""
        self.table_data[table_name]['operations'] = operations

    def write_ops_to_aws(self):
        """Write all the operations to AWS"""
        for name, table_info in self.table_data.items():
            wr.catalog.upsert_table_parameters(parameters={'operations': json.dumps(table_info['operations'])},
                                               database=self.database,
                                               table=name)

    def set_table_tags_ops(self, table_name: str, tags: list, operations: list):
        """Set the tags and operations for a specific table"""
        self.set_table_tags(table_name, tags)
        self.set_table_ops(table_name, operations)

    def get_table_name_from_file(self, file_name: str) -> [str, None]:
        """Set the tags and operations for a specific table"""
        for name, table_info in self.table_data.items():
            if table_info['location'] == file_name:
                return name

        # We didn't find it
        return None


if __name__ == '__main__':
    from pprint import pprint

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str, default='all', help='Only show a specific database')
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print('Unrecognized args: %s' % commands)
        sys.exit(1)

    # Create the class and get the AWS Glue database info
    glue_meta = GlueMeta(args.database)

    # List files that got ignored
    print('Ignored Files:')
    for my_name in glue_meta.ignored_files:
        print(f"\t{my_name}")

    # Get a specific table
    my_table = 'udm_com_idya_prj_a1_csv'
    my_table_info = glue_meta.get_table_info(my_table)
    pprint(my_table_info)

    # Set the tags and operations for this table
    glue_meta.set_table_tags(my_table, tags=['tag1', 'tag2'])
    my_table_info = glue_meta.get_table_info(my_table)
    pprint(my_table_info)

    # Test out getting just the tables with certain tags
    for my_table in glue_meta.tables_with_tags(['assay', 'partial']):
        print(my_table)

    # Test out include/exclude tags
    for my_table in glue_meta.tables_with_tags(['smiles']):
        print(my_table)
    print('\nNO PROJECT')
    for my_table in glue_meta.tables_with_tags(['smiles'], exclude_list=['project', 'reagent']):
        print(my_table)

    # Test out column name retrieval
    my_columns = glue_meta.get_table_columns(my_table)

    # Test out number of column
    print(f'Number of columns {len(my_columns)}')

    # Test out both the glue computed and registered types
    test_table = 'udm_com_idya_prj_a1_csv'
    my_table_info = glue_meta.get_table_info(test_table)
    pprint(my_table_info)
    print(my_table_info['glue_computed_types'])

    # When you ask for dtypes two S3 calls are made (slow)
    # Note: Lazy evaluation here.. so a second call is fast
    my_dtypes = glue_meta.get_column_dtypes(test_table)
    pprint(my_dtypes)

    # This call will be fast
    my_dtypes = glue_meta.get_column_dtypes(test_table)
    pprint(my_dtypes)
