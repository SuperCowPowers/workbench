"""DataSource: Abstract Base Class for all data sources (S3: CSV, Parquet, RDS, etc)"""
from abc import ABC, abstractmethod
import pandas as pd
import awswrangler as wr


class DataSource(ABC):
    """DataSource: Abstract Base Class for all data sources (S3: CSV, Parquet, RDS, etc)"""
    def __init__(self, name, resource_url: str):
        self.name = name
        self.resource_url = resource_url
        self.data_catalog_db = 'sageworks'
        self.num_rows = None
        self.num_columns = None
        self.column_names = None
        self.tags = []

        # Make sure the AWS data catalog database exists
        self.ensure_aws_catalog_db()

        # All done
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

    @abstractmethod
    def generate_feature_set(self, feature_type: str) -> bool:
        """Concrete Classes will support different feature set generations"""
        pass

    def add_tag(self, tag):
        """Add a tag to this data source"""
        # This ensures no duplicate tags
        self.tags = list(set(self.tags).union([tag]))

    def ensure_aws_catalog_db(self):
        """Ensure that the AWS Catalog Database exists"""
        wr.catalog.create_database(self.data_catalog_db, exist_ok=True)
