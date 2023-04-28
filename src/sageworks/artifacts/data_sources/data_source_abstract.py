"""DataSourceAbstract: Abstract Base Class for all data sources (S3: CSV, JSONL, Parquet, RDS, etc)"""
from abc import abstractmethod
import pandas as pd

# Local imports
from sageworks.artifacts.artifact import Artifact


class DataSourceAbstract(Artifact):
    def __init__(self, uuid):
        """DataSourceAbstract: Abstract Base Class for all data sources (S3: CSV, JSONL, Parquet, RDS, etc)"""

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
    def column_types(self) -> list[str]:
        """Return the column types for this Data Source"""
        pass

    def column_details(self) -> dict:
        """Return the column details for this Data Source"""
        return {name: type_ for name, type_ in zip(self.column_names(), self.column_types())}

    @abstractmethod
    def query(self, query: str) -> pd.DataFrame:
        """Query the DataSourceAbstract"""
        pass

    @abstractmethod
    def sample_df(self, max_rows: int = 1000) -> pd.DataFrame:
        """Return a sample DataFrame from this DataSource"""
        pass

    def details(self) -> dict:
        """Additional Details about this DataSourceAbstract Artifact"""
        details = self.summary()
        details["num_rows"] = self.num_rows()
        details["num_columns"] = self.num_columns()
        details["column_details"] = self.column_details()
        return details
