"""S3CSV: Class for an S3 CSV DataSource"""
from abc import ABC, abstractmethod
import pandas as pd

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
        return True

    def query(self, query: str) -> pd.DataFrame:
        """Query the S3 CSV file"""
        print(f'Query {query}...')
        return pd.DataFrame()

    def generate_feature_set(self, feature_type: str) -> bool:
        """Concrete Classes will support different feature set generations"""
        print(f'Generating feature set {feature_type}...')
        return True


# Simple test of the S3 CSV functionality
def test():
    """Test for S3 CSV Class"""

    # Create a Data Source
    my_data = S3CSV('aqsol_data', 's3://scp-sageworks-incoming-data/aqsol_public_data.csv')

    # Call the various methods
    assert(my_data.check())
    df = my_data.query('some query')
    print(df)
    my_data.generate_feature_set('rdkit')
    my_data.save_meta()
    meta = my_data.get_meta()
    print(meta)


if __name__ == "__main__":
    test()
