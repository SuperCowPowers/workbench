"""DataSource: Abstract Base Class for all data sources (S3: CSV, Parquet, RDS, etc)"""
from abc import abstractmethod
import pandas as pd

# Local imports
from sageworks.artifacts.artifact import Artifact


class DataSource(Artifact):
    def __init__(self, uuid):
        """DataSource: Abstract Base Class for all data sources (S3: CSV, Parquet, RDS, etc)"""

        # Call superclass init
        super().__init__(uuid)

    @abstractmethod
    def num_rows(self) -> int:
        """Return the number of rows for this Data Source"""
        pass

    @abstractmethod
    def num_columns(self) -> int:
        """Return the number of columns for this Data Source"""
        pass

    @abstractmethod
    def column_names(self) -> list[str]:
        """Return the column names for this Data Source"""
        pass

    @abstractmethod
    def query(self, query: str) -> pd.DataFrame:
        """Query the DataSource"""
        pass

    def details(self) -> dict:
        """Additional Details about this DataSource Artifact"""
        details = self.info()
        details["num_rows"] = self.num_rows()
        details["num_columns"] = self.num_columns()
        details["column_names"] = self.column_names()
        return details
