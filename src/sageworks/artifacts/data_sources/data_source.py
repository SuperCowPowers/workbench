"""DataSource: Abstract Base Class for all data sources (S3: CSV, Parquet, RDS, etc)"""
from abc import abstractmethod
import pandas as pd
import awswrangler as wr

# Local imports
from sageworks.artifacts.artifact import Artifact


class DataSource(Artifact):
    def __init__(self):
        """DataSource: Abstract Base Class for all data sources (S3: CSV, Parquet, RDS, etc)"""

        # Call superclass init
        super().__init__()

        # Make sure the AWS data catalog database exists
        self.ensure_aws_catalog_db()

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

    def ensure_aws_catalog_db(self):
        """Ensure that the AWS Catalog Database exists"""
        wr.catalog.create_database(self.data_catalog_db, exist_ok=True)
