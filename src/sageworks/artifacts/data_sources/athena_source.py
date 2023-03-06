"""AthenaSource: SageWork Data Source accessible through Athena"""
import pandas as pd
import awswrangler as wr
import logging
from datetime import datetime

# Local Imports
from sageworks.artifacts.data_sources.data_source import DataSource
from sageworks.meta.sageworks_meta import MetaCategory, SageWorksMeta
from sageworks.utils.logging import logging_setup

# Setup Logging
logging_setup()


class AthenaSource(DataSource):
    """AthenaSource: SageWork Data Source accessible through Athena"""
    def __init__(self, database_name, table_name):
        """AthenaSource Initialization

        Args:
            database_name (str): Name of Athena Database
            table_name (str): Name of Athena Table
        """

        self.database_name = database_name
        self.table_name = table_name
        self.log = logging.getLogger(__name__)

        # Call SuperClass (DataSource) Initialization
        super().__init__(self.database_name)

        # Grab a SageWorks Metadata object and pull information for Data Sources
        self.sage_meta = SageWorksMeta()
        self._meta = self.sage_meta.get_metadata(MetaCategory.DATA_SOURCES)[self.database_name].get(self.table_name)

        # Grab my tags
        self._tags = self._meta['Parameters']['tags']

        # All done
        print(f'AthenaSource Initialized: {self.database_name}.{self.table_name}')

    def check(self) -> bool:
        """Does the table_name exist in the SageWorks Metadata?"""
        if self._meta is None:
            self.log.critical(f'AthenaSource.check() {self.table_name} not found in SageWorks Metadata!')
            return False
        return True

    def size(self) -> bool:
        """Return the size of this data in MegaBytes"""
        # FIXME: WAG for now, figure out better way to get this
        data_bytes = self.num_rows() * self.num_columns() * 20
        megabytes = int(data_bytes / 1_000_000)
        return megabytes

    def meta(self):
        """Get the metadata for this artifact"""
        return self._meta

    def tags(self):
        """Get the tags for this artifact"""
        return self._tags

    def add_tag(self, tag):
        """Add a tag to this artifact"""
        # This ensures no duplicate tags
        self._tags = list(set(self._tags).union([tag]))

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self._meta['CreateTime']

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        return self._meta['UpdateTime']

    def num_rows(self) -> int:
        """Return the number of rows for this Data Source"""
        count_df = self.query(f'select count(*) AS count from "{self.database_name}"."{self.table_name}"')
        return count_df['count'][0]

    def num_columns(self) -> int:
        """Return the number of columns for this Data Source"""
        return len(self.column_names())

    def column_names(self) -> list[str]:
        """Return the column names for this Athena Table"""
        return [item['Name'] for item in self._meta['StorageDescriptor']['Columns']]

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


# Simple test of the AthenaSource functionality
def test():
    """Test for AthenaSource Class"""

    # Retrieve a Data Source
    my_data = AthenaSource('sageworks', 'aqsol_data')

    # Call the various methods

    # Does this Athena data exist?
    assert(my_data.check())

    # What's the size of the data?
    print(f"Size of Data (MB): {my_data.size()}")

    # When was it created and last modified?
    print(f"Created: {my_data.created()}")
    print(f"Modified: {my_data.modified()}")

    # How many rows and columns?
    num_rows = my_data.num_rows()
    num_columns = my_data.num_columns()
    print(f'Rows: {num_rows} Columns: {num_columns}')

    # What are the column names?
    columns = my_data.column_names()
    print(columns)

    # Get all the metadata associated with this data source
    print(f"Meta: {my_data.meta()}")

    # Get the tags associated with this data source
    print(f"Tags: {my_data.tags()}")

    # Run a query to only pull back a few columns and rows
    column_query = ', '.join(columns[:3])
    query = f'select {column_query} from "{my_data.database_name}"."{my_data.table_name}" limit 10'
    df = my_data.query(query)
    print(df)


if __name__ == "__main__":
    test()
