"""DataSource: Abstract Base Class for all data sources (S3: CSV, Parquet, RDS, etc)"""
from abc import abstractmethod
import pandas as pd
import awswrangler as wr

# Local imports
from sageworks.aws_artifacts.aws_artifact import AWSArtifact


class DataSource(AWSArtifact):
    def __init__(self, data_catalog_db: str = 'sageworks'):
        """DataSource: Abstract Base Class for all data sources (S3: CSV, Parquet, RDS, etc)

        Args:
            data_catalog_db (str): AWS Data Catalog Database. Defaults to 'sageworks'.
        """

        # Initialize the data source attributes
        self.data_catalog_db = data_catalog_db

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
