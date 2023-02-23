"""DataSource: Abstract Base Class for all data sources (S3: CSV, Parquet, RDS, etc)"""
from abc import ABC, abstractmethod
import pandas as pd

# Local Imports


class DataSource(ABC):
    """DataSource: Abstract Base Class for all data sources (S3: CSV, Parquet, RDS, etc)"""
    def __init__(self, name, resource_url: str):
        self.name = name
        self.resource_url = resource_url
        self.meta = {
            'name': name,
            'resource_url': resource_url
        }
        print(f'DataSource Initialized: {resource_url}')

    @abstractmethod
    def check(self) -> bool:
        """Does the DataSource exist? Can we connect to it?"""
        print(f'Checking {self.name}...')

    @abstractmethod
    def query(self, query: str) -> pd.DataFrame:
        """Query the DataSource"""
        print(f'Query {query}...')
        return pd.DataFrame()

    @abstractmethod
    def generate_feature_set(self, feature_type: str) -> bool:
        """Concrete Classes will support different feature set generations"""
        print(f'Generating feature set {feature_type}...')
        return True

    def save_meta(self) -> bool:
        print('Saving to meta-data store...')
        print(self.meta)
        return True

    def get_meta(self) -> dict:
        return self.meta
