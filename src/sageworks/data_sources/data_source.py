"""DataSource: Abstract Base Class for all data sources (S3: CSV, Parquet, RDS, etc)"""
from abc import ABC, abstractmethod
import pandas as pd
# Local Imports


class DataSource(ABC):
    """DataSource: Abstract Base Class for all data sources (S3: CSV, Parquet, RDS, etc)"""
    def __init__(self, name, resource_url: str):
        self.name = name
        self.resource_url = resource_url
        self.num_rows = None
        self.num_columns = None
        self.column_names = None
        print(f'DataSource Initialized: {resource_url}')

    @abstractmethod
    def check(self) -> bool:
        """Does the DataSource exist? Can we connect to it?"""
        pass

    @abstractmethod
    def get_num_rows(self) -> int:
        """Return the number of rows for this Data Source"""
        pass

    @abstractmethod
    def get_num_columns(self) -> int:
        """Return the number of columns for this Data Source"""
        pass

    @abstractmethod
    def get_column_names(self) -> list[str]:
        """Return the column names for this Data Source"""
        pass

    @abstractmethod
    def query(self, query: str) -> pd.DataFrame:
        """Query the DataSource"""
        pass

    @abstractmethod
    def generate_feature_set(self, feature_type: str) -> bool:
        """Concrete Classes will support different feature set generations"""
        pass

    def store_meta(self) -> bool:
        print('Storing meta-data...')
        return True

    def get_meta(self) -> dict:
        return vars(self)
