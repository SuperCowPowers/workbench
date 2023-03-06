"""DataSource: Abstract Base Class for all data sources (S3: CSV, Parquet, RDS, etc)"""
from abc import abstractmethod
import pandas as pd
import awswrangler as wr

# Local imports
from sageworks.artifacts.artifact import Artifact


class DataSource(Artifact):
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

    @abstractmethod
    def schema_validation(self):
        """All data sources will have some form of schema validation.
           For things like RDS tables this might just be a return True
           but for other data source types there should be some code that
           looks at the columns and types and does something reasonable
           """
        pass

    @abstractmethod
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
        pass

    def ensure_aws_catalog_db(self):
        """Ensure that the AWS Catalog Database exists"""
        wr.catalog.create_database(self.data_catalog_db, exist_ok=True)
