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
    def sample_df(self, recompute: bool = False) -> pd.DataFrame:
        """Return a sample DataFrame from this DataSource
        Args:
            recompute(bool): Recompute the sample (default: False)
        Returns:
            pd.DataFrame: A sample DataFrame from this DataSource
        """
        pass

    @abstractmethod
    def quartiles(self, recompute: bool = False) -> dict[dict]:
        """Compute Quartiles for all the numeric columns in a DataSource
        Args:
            recompute(bool): Recompute the quartiles (default: False)
        Returns:
            dict(dict): A dictionary of quartiles for each column in the form
                 {'col1': {'min': 0, 'q1': 1, 'median': 2, 'q3': 3, 'max': 4},
                  'col2': ...}
        """
        pass

    def details(self) -> dict:
        """Additional Details about this DataSourceAbstract Artifact"""
        details = self.summary()
        details["num_rows"] = self.num_rows()
        details["num_columns"] = self.num_columns()
        details["column_details"] = self.column_details()
        details.update(self.meta())
        return details
