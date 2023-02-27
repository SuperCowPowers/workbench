"""AthenaSource: SageWork Data Source accessible through Athena"""
import pandas as pd
import awswrangler as wr
import logging

# Local Imports
from sageworks.data_sources.data_source import DataSource
from sageworks.meta.sageworks_meta import MetaCategory, SageWorksMeta
from sageworks.utils.logging import logging_setup

# Setup Logging
logging_setup()


class AthenaSource(DataSource):
    """AthenaSource: SageWork Data Source accessible through Athena"""
    def __init__(self, table_name):
        """AthenaSource Initialization

        Args:
            table_name (str): Name of Athena Table in the SageWorks Database
        """
        self.table_name = table_name
        self.log = logging.getLogger(__name__)

        # Call SuperClass (DataSource) Initialization
        super().__init__(table_name, resource_url=None)

        # Grab a SageWorks Metadata object and pull information for Data Sources
        self.sage_meta = SageWorksMeta()
        self.my_meta = self.sage_meta.get(MetaCategory.DATA_SOURCES).get(self.table_name)

        # All done
        print(f'AthenaSource Initialized: {table_name}')

    def check(self) -> bool:
        """Does the table_name exist in the SageWorks Metadata?"""
        if self.my_meta is None:
            self.log.critical(f'AthenaSource.check() {self.table_name} not found in SageWorks Metadata!')
            return False
        return True

    def get_num_rows(self) -> int:
        """Return the number of rows for this Data Source"""
        count_df = self.query(f"select count(*) AS count from {self.table_name}")
        return count_df['count'][0]

    def get_num_columns(self) -> int:
        """Return the number of columns for this Data Source"""
        if self.column_names is None:
            self.get_column_names()
        self.num_columns = len(self.column_names)
        return self.num_columns

    def get_column_names(self) -> list[str]:
        """Return the column names for this Athena Table"""
        if self.column_names is None:
            self.column_names = [item['Name'] for item in self.my_meta['StorageDescriptor']['Columns']]
        return self.column_names

    def query(self, query: str) -> pd.DataFrame:
        """Query the AthenaSource"""
        df = wr.athena.read_sql_query(sql=query, database=self.data_catalog_db)
        scanned_bytes = df.query_metadata["Statistics"]["DataScannedInBytes"]
        self.log.info(f"Athena Query successful (scanned bytes: {scanned_bytes})")
        return df

    def schema_validation(self):
        """All data sources will have some form of schema validation.
           For things like RDS tables this might just be a return True
           but for other data source types there should be some code that
           looks at the columns and types and does something reasonable
           """
        return True

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
        return True

    def generate_feature_set(self, feature_type: str) -> bool:
        """Concrete Classes will support different feature set generations"""
        return True


# Simple test of the AthenaSource functionality
def test():
    """Test for AthenaSource Class"""

    # Create a Data Source
    my_data = AthenaSource('aqsol_data')

    # Call the various methods

    # Does this Athena data exist?
    assert(my_data.check())

    # How many rows and columns?
    rows = my_data.get_num_rows()
    columns = my_data.get_num_columns()
    print(f'Rows: {rows} Columns: {columns}')

    # What are the column names?
    print(my_data.get_column_names())

    # Run a query to only pull back only some data
    query = f"select id, name, smiles from {my_data.table_name} limit 10"
    df = my_data.query(query)
    print(df)

    # Generate a feature set from the data source
    my_data.generate_feature_set('rdkit')


if __name__ == "__main__":
    test()
