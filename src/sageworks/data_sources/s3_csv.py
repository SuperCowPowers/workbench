"""S3CSV: Class for an S3 CSV DataSource"""
import pandas as pd
import awswrangler as wr
import json

# Local Imports
from sageworks.data_sources.data_source import DataSource


class S3CSV(DataSource):
    """"S3CSV: Class for an S3 CSV DataSource"""
    def __init__(self, name, resource_url: str):

        # S3 path if they want to store this data
        self.data_source_s3_path = 's3://sageworks-data-sources'

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

    def schema_validation(self):
        """All data sources will have some form of schema validation.
           For things like RDS tables this might just be a return True
           but for other data source types there should be some code that
           looks at the columns and types and does something reasonable
           """
        print('Schema Validation...')
        return True

    def data_quality_review(self):
        """All data sources should have some form of data quality review.
           After schema_validation the class should 'inspect' the data values.
           - What percentage of values in each column are NaNs/Null?
           - Are there only a few unique values in each column (low variance?)
           - Are there 'crazy' values way out of 3Sigma range?
           - TBD
           Given that plots and charts will be super helpful for this the data
           quality review process should probably involve a web page and click 'OK' button
           Having a Data Quality Web page is something we want to do anyway
           """
        print('Data Quality Review...')
        return True

    def _store_in_sageworks(self, overwrite=True):
        """Convert the CSV data into Parquet Format, move it to SageMaker S3 Path, and
           the information about the data to the AWS Data Catalog sageworks database"""
        s3_storage_path = f"{self.data_source_s3_path}/{self.name}"
        print('Storing into SageWorks...')

        # Specify the tags to store
        self.add_tag('sageworks')
        self.add_tag('public')

        # FIXME: This pull all the data down to the client (improve this later)
        df = wr.s3.read_csv(self.resource_url, low_memory=False)
        wr.s3.to_parquet(df, path=s3_storage_path, dataset=True, mode='overwrite',
                         database=self.data_catalog_db, table=self.name,
                         description=f'SageWorks data source: {self.name}',
                         filename_prefix=f'{self.name}_',
                         parameters={'tags': json.dumps(self.tags)},
                         partition_cols=None)  # FIXME: Have some logic around partition columns

    def generate_feature_set(self, feature_type: str) -> bool:
        """Concrete Classes will support different feature set generations"""
        print(f'Generating feature set {feature_type}...')
        return True


# Simple test of the S3 CSV functionality
def test():
    """Test for S3 CSV Class"""
    from pprint import pprint

    # Create a Data Source
    my_data = S3CSV('aqsol_data', 's3://sageworks-incoming-data/aqsol_public_data.csv')

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

    # Get the meta-data for our data source
    meta = my_data.get_meta()
    pprint(meta)

    # Store this data source into SageWorks also puts info in AWS Data Catalog
    my_data.store_in_sageworks()


if __name__ == "__main__":
    test()
