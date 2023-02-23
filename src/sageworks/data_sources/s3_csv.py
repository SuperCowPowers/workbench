"""S3CSV: Class for an S3 CSV DataSource"""
import pandas as pd
import awswrangler as wr

# Local Imports
from sageworks.data_sources.data_source import DataSource


class S3CSV(DataSource):
    """"S3CSV: Class for an S3 CSV DataSource"""
    def __init__(self, name, resource_url: str):

        # Call Base Class Initialization
        super().__init__(name, resource_url)

    def check(self) -> bool:
        """Does the S3 CSV exist"""
        print(f'Checking {self.name}...')
        return wr.s3.does_object_exist(self.resource_url)

    def get_num_rows(self) -> int:
        """Return the number of rows for this Data Source"""
        # Okay the count df seems to be broken up into fractions that we can sum up
        # Note: This probably needs a deeper dive/understanding at some point
        count_df = self.query('select count(*) from s3object')
        self.num_rows = count_df.sum().values[0]
        return self.num_rows

    def get_num_columns(self) -> int:
        """Return the number of columns for this Data Source"""
        if self.column_names is None:
            self.get_column_names()
        self.num_columns = len(self.column_names)
        return self.num_columns

    def get_column_names(self) -> list[str]:
        """Return the column names for this CSV File"""
        if self.column_names is None:
            self.column_names = list(self.query('select * from s3object limit 1').columns)
        return self.column_names

    def query(self, query: str) -> pd.DataFrame:
        """Query the S3 CSV file"""
        print(f'Query {query}...')
        params = {'FileHeaderInfo': 'Use'}
        df = wr.s3.select_query(query, self.resource_url, input_serialization='CSV', input_serialization_params=params)
        return df

    def generate_feature_set(self, feature_type: str) -> bool:
        """Concrete Classes will support different feature set generations"""
        print(f'Generating feature set {feature_type}...')
        return True


# Simple test of the S3 CSV functionality
def test():
    """Test for S3 CSV Class"""
    from pprint import pprint

    # Create a Data Source
    my_data = S3CSV('aqsol_data', 's3://scp-sageworks-incoming-data/aqsol_public_data.csv')

    # Call the various methods

    # Does my S3 data exist?
    assert(my_data.check())

    # How many rows and columns?
    rows = my_data.get_num_rows()
    columns = my_data.get_num_columns()
    print(f'Rows: {rows} Columns: {columns}')

    # What are the column names?
    print(my_data.get_column_names())

    # Run a query to only pull back only some of the data
    query = 'select ID, Name, SMILES from s3object'
    df = my_data.query(query)
    print(df)

    # Generate a feature set from the data source
    my_data.generate_feature_set('rdkit')

    # Store meta-data to the meta-data DB and retrieve it
    my_data.store_meta()
    meta = my_data.get_meta()
    pprint(meta)


if __name__ == "__main__":
    test()
